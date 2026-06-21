"""MLA retrieval-pattern analyzer (Phase 3 / Story 5 re-scoped).

Story 5's original premise ("trace DSA indexer top-K selection") was
empirically rejected on 2026-06-20 — GLM-5.2 in stock llama.cpp runs as
plain absorbed MLA via the ``is_lite=false`` branch in ``deepseek2::graph``;
the ``indexer_*`` weights are loaded but never invoked. The DSA lightning
indexer is a non-firing code path for this model. See
``GLM52_SESSION_MEMORY.md`` → "Phase 3 / Story 5 DSA forward-path patch —
EMPIRICALLY REJECTED" for the full forensic record.

The re-scoped goal: trace MLA's *actual* retrieval mechanism (full attention
over the KV cache) using the existing ``activation_summary`` event type on
``q_nope_absorbed`` (the absorbed query in MLA ~ what each generation-step
token "asks for") and ``kv_cmpr`` (the lora-compressed KV ~ what each prefill
token "offers"). For each generation-step query, score every earlier prefill
position by signed top-K channel overlap (sum over shared channels of
q_mag * k_mag); the top-N scores approximate "which earlier positions the
model retrieves from" when generating that token.

This is an APPROXIMATION of softmax(QK) attention — we only have the top-K
channel magnitudes per (token, layer), not full vectors. Top-K magnitude
overlap is a defensible interpretability lower-bound: it surfaces positions
whose dominant latent dimensions align with the current query's dominant
dimensions. Document this limitation in every report.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .schema import ActivationSummaryRecord, PHASE_GENERATION, PHASE_PREFILL

DEFAULT_Q_STEM = "q_nope_absorbed"
DEFAULT_K_STEM = "kv_cmpr"
DEFAULT_RETRIEVAL_TOPN = 10

# Distance bucket cutoffs as fractions of prompt length. Absolute floor on
# "recent" so a 10-token prompt doesn't bucket everything as recent.
RECENT_FRACTION = 0.05      # within 5% of context
MEDIUM_FRACTION = 0.30      # 5%–30%
FAR_FRACTION = 0.70         # 30%–70%
RECENT_FLOOR = 64           # also bucket anything within 64 tokens as recent

BUCKET_RECENT = "recent"
BUCKET_MEDIUM = "medium"
BUCKET_FAR = "far"
BUCKET_VERY_FAR = "very_far"
BUCKET_FUTURE = "future"
BUCKET_ORDER = (BUCKET_RECENT, BUCKET_MEDIUM, BUCKET_FAR, BUCKET_VERY_FAR, BUCKET_FUTURE)


@dataclass
class RetrievedPosition:
    """One retrieved position for one (query, layer) pair."""

    position: int
    score: float           # signed overlap: sum over shared channels of q*c * k*c
    shared_channels: int  # count of channels shared in top-K(Q) ∩ top-K(K)
    distance: int         # tokens between query position and retrieved position


@dataclass
class RetrievalResult:
    """Top-N retrieved positions for a single (generation-step query, layer)."""

    run_id: str
    layer: int
    query_token: int
    task_label: str
    language: str
    prompt_len: int
    top_positions: list[RetrievedPosition] = field(default_factory=list)


@dataclass
class RetrievalAnalysis:
    """Aggregated retrieval-pattern analysis across all queries/layers."""

    results: list[RetrievalResult] = field(default_factory=list)
    bucket_counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(BUCKET_ORDER, 0))
    sentinel_hits: int = 0      # (query, layer) pairs whose top-N included ≥1 sentinel pos
    sentinel_total: int = 0     # total (query, layer) pairs evaluated for sentinel
    q_stem: str = ""
    k_stem: str = ""
    topn: int = 0
    sentinel_range: Optional[tuple[int, int]] = None
    n_query_records: int = 0
    n_key_records: int = 0  # includes prefill + generation k records total


def signed_overlap(
    q_top: list[list[float]],
    k_top: list[list[float]],
) -> tuple[float, int]:
    """Signed dot-product overlap over top-K(Q) ∩ top-K(K).

    Returns (score, n_shared). For each channel appearing in BOTH top-K lists,
    add q_mag * k_mag (signed product). The summed score is a lower bound on the
    true dot product q·k restricted to the shared channels; values outside top-K
    aren't materialized by the tracer, so this is the best we can do.
    """
    # Build channel→magnitude maps (last write wins; top_k_channels are unique
    # per record by construction in the C++ tracer).
    q_map: dict[int, float] = {int(c): float(m) for c, m in q_top}
    k_map: dict[int, float] = {int(c): float(m) for c, m in k_top}
    shared = q_map.keys() & k_map.keys()
    score = 0.0
    for c in shared:
        score += q_map[c] * k_map[c]
    return score, len(shared)


def distance_bucket(query_pos: int, retrieved_pos: int, prompt_len: int) -> str:
    """Map distance between query position and a retrieved earlier position to a bucket.

    Buckets are defined relative to the prompt length (so 20k-token prompts and
    30-token prompts both produce interpretable distance distributions):

    - ``recent``: within 5% of prompt_len, or within RECENT_FLOOR tokens
    - ``medium``: 5%–30%
    - ``far``: 30%–70%
    - ``very_far``: 70%–100% (start of prompt)
    - ``future``: retrieved_pos >= query_pos (shouldn't normally happen)
    """
    if retrieved_pos >= query_pos:
        return BUCKET_FUTURE
    d = query_pos - retrieved_pos
    if d <= RECENT_FLOOR:
        return BUCKET_RECENT
    if prompt_len <= 0:
        # No prompt-length signal (e.g., all-prefill test); fall back to
        # absolute distance thresholds.
        if d <= 256:
            return BUCKET_RECENT
        if d <= 2048:
            return BUCKET_MEDIUM
        if d <= 8192:
            return BUCKET_FAR
        return BUCKET_VERY_FAR
    if d <= prompt_len * RECENT_FRACTION:
        return BUCKET_RECENT
    if d <= prompt_len * MEDIUM_FRACTION:
        return BUCKET_MEDIUM
    if d <= prompt_len * FAR_FRACTION:
        return BUCKET_FAR
    return BUCKET_VERY_FAR


def analyze_retrieval(
    activation_records: Iterable[ActivationSummaryRecord],
    *,
    q_stem: str = DEFAULT_Q_STEM,
    k_stem: str = DEFAULT_K_STEM,
    topn: int = DEFAULT_RETRIEVAL_TOPN,
    sentinel_range: Optional[tuple[int, int]] = None,
) -> RetrievalAnalysis:
    """Compute approximate MLA retrieval patterns from q+k activation summaries.

    For each generation-step ``q_stem`` record at (token=q_pos, layer=L):

    - Find all ``k_stem`` records at earlier positions `(token_index < q_pos)`
      at the same (run_id, layer) — the prefill+previous-generation KV set.
    - Score each candidate position by signed top-K channel overlap
      (sum over shared channels of q_mag * k_mag).
    - Take top-N scored positions → ``top_positions``.

    Aggregates across all (query, layer) pairs:

    - ``bucket_counts``: distribution of retrieved positions into
      recent/medium/far/very_far buckets by distance from the query.
    - ``sentinel_hits`` / ``sentinel_total``: if ``sentinel_range=(start, end)``
      is provided, count how many (query, layer) pairs have at least one
      retrieved position inside ``[start, end]``.

    Returns the populated :class:`RetrievalAnalysis`. If no q+k records match
    the requested stems, ``results`` is empty (caller can render empty-section
    or skip).
    """
    # Index k_stem records by (run_id, layer, token_index). Last write per
    # (run, layer, token) wins — duplicate emissions for the same strides
    # (kv_cmpr fires as both the matmul output and the cache write) collapse.
    k_index: dict[tuple[str, int, int], ActivationSummaryRecord] = {}
    q_records: list[ActivationSummaryRecord] = []
    # Track per-run prompt length (~max prefill token_index + 1).
    run_prompt_len: dict[str, int] = {}

    n_q_total = 0
    n_k_total = 0
    for rec in activation_records:
        if rec.tensor_stem == k_stem:
            n_k_total += 1
            # Index every k record (prefill + generation). Only the query path
            # checks phase; the key index can include same-run future keys
            # (we filter by reversed position at scoring time).
            k_index[(rec.run_id, rec.layer, rec.token_index)] = rec
            if rec.phase == PHASE_PREFILL:
                cur = run_prompt_len.get(rec.run_id, 0)
                if rec.token_index + 1 > cur:
                    run_prompt_len[rec.run_id] = rec.token_index + 1
        elif rec.tensor_stem == q_stem:
            n_q_total += 1
            # Only score generation-step queries — prefill queries aren't
            # "retrieval" (they're encoding). Generation step q records are
            # the actual "what does the model ask for at this token".
            if rec.phase == PHASE_GENERATION:
                q_records.append(rec)

    analysis = RetrievalAnalysis(
        q_stem=q_stem,
        k_stem=k_stem,
        topn=topn,
        sentinel_range=sentinel_range,
        n_query_records=n_q_total,
        n_key_records=n_k_total,
    )

    # Per-(run, layer): a small index of (token_index → record) so we don't
    # re-scan the global dict. Building this lazily per query is O(1) lookup
    # in a dict[run_layer] -> {token: record}.
    run_layer_keys: dict[tuple[str, int], dict[int, ActivationSummaryRecord]] = defaultdict(dict)
    for (run_id, layer, tok), k_rec in k_index.items():
        run_layer_keys[(run_id, layer)][tok] = k_rec

    for q_rec in q_records:
        prompt_len = run_prompt_len.get(q_rec.run_id, 0)
        key_map = run_layer_keys.get((q_rec.run_id, q_rec.layer), {})
        q_pos = q_rec.token_index
        scored: list[tuple[float, int, int]] = []
        for k_pos, k_rec in key_map.items():
            if k_pos >= q_pos:
                continue
            score, shared = signed_overlap(q_rec.top_k_channels, k_rec.top_k_channels)
            scored.append((score, k_pos, shared))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:topn]
        retrieved = [
            RetrievedPosition(
                position=k_pos,
                score=score,
                shared_channels=shared,
                distance=q_pos - k_pos,
            )
            for score, k_pos, shared in top
        ]
        result = RetrievalResult(
            run_id=q_rec.run_id,
            layer=q_rec.layer,
            query_token=q_pos,
            task_label=q_rec.task_label,
            language=q_rec.language,
            prompt_len=prompt_len,
            top_positions=retrieved,
        )
        analysis.results.append(result)

        for rp in retrieved:
            b = distance_bucket(q_pos, rp.position, prompt_len)
            analysis.bucket_counts[b] = analysis.bucket_counts.get(b, 0) + 1

        if sentinel_range is not None:
            s_start, s_end = sentinel_range
            analysis.sentinel_total += 1
            # Even queries with empty top_positions count toward sentinel_total:
            # the query was scored (just against zero candidates) — recording
            # this misses differently. Avoids inflating hit_rate.
            for rp in retrieved:
                if s_start <= rp.position <= s_end:
                    analysis.sentinel_hits += 1
                    break

    return analysis


def to_summary_dict(analysis: RetrievalAnalysis) -> dict[str, Any]:
    """Serialize a :class:`RetrievalAnalysis` into a JSON-stable dict.

    Stable key ordering (sorted) so the SHA-256 hash of the parent summary
    (computed in :mod:`analyze`) is deterministic regardless of dict insertion.
    """
    total_bucket = sum(analysis.bucket_counts.values())
    return {
        "q_stem": analysis.q_stem,
        "k_stem": analysis.k_stem,
        "topn": analysis.topn,
        "sentinel_range": (
            list(analysis.sentinel_range) if analysis.sentinel_range is not None else None
        ),
        "n_query_records_seen": analysis.n_query_records,
        "n_key_records_seen": analysis.n_key_records,
        "n_results": len(analysis.results),
        "bucket_counts": {b: analysis.bucket_counts.get(b, 0) for b in BUCKET_ORDER},
        "bucket_fractions": {
            b: (analysis.bucket_counts.get(b, 0) / total_bucket if total_bucket else 0.0)
            for b in BUCKET_ORDER
        },
        "sentinel_hits": analysis.sentinel_hits,
        "sentinel_total": analysis.sentinel_total,
        "sentinel_hit_rate": (
            analysis.sentinel_hits / analysis.sentinel_total
            if analysis.sentinel_total > 0
            else None
        ),
        "sample_results": [
            {
                "run_id": r.run_id,
                "layer": r.layer,
                "query_token": r.query_token,
                "task_label": r.task_label,
                "language": r.language,
                "prompt_len": r.prompt_len,
                "top_positions": [
                    {
                        "position": rp.position,
                        "score": rp.score,
                        "shared_channels": rp.shared_channels,
                        "distance": rp.distance,
                    }
                    for rp in r.top_positions
                ],
            }
            # Latest N results only — full list could be large for many-layer runs
            for r in analysis.results[-20:]
        ],
    }


def render_markdown(analysis: RetrievalAnalysis, *, top_results: int = 20) -> str:
    """Render the MLA retrieval analysis section as markdown.

    Returns ``""`` if ``analysis.results`` is empty — the caller is expected to
    check and skip the section entirely in that case (no empty header).
    """
    if not analysis.results:
        return ""

    lines: list[str] = []
    lines.append("## MLA retrieval analysis (Phase 3 / Story 5 re-scoped)")
    lines.append("")
    lines.append(
        f"Approximate MLA-latent-attention retrieval from `{analysis.q_stem}` queries "
        f"and `{analysis.k_stem}` keys (captured via `--trace-activations "
        f"{analysis.q_stem},{analysis.k_stem}`). For each generation-step query "
        f"(token, layer), scored every earlier prefill position by signed top-K "
        f"channel overlap (Σ over shared channels of q·c × k·c), ranked top-N "
        f"positions by descending score."
    )
    lines.append("")
    lines.append(
        f"- Query records seen: **{analysis.n_query_records}** "
        f"(generation-step only — prefill queries aren't retrieval)"
    )
    lines.append(f"- Key records seen: **{analysis.n_key_records}** (prefill + gen keys)")
    lines.append(f"- (query, layer) pairs scored: **{len(analysis.results)}**")
    if analysis.sentinel_range is not None:
        lines.append(f"- Sentinel position range: **`{analysis.sentinel_range[0]}..{analysis.sentinel_range[1]}`** (tokenized)")
    lines.append("")

    # Distance-bucket distribution
    total = sum(analysis.bucket_counts.values())
    if total > 0:
        lines.append("### Distance buckets (all retrieved positions)")
        lines.append("")
        lines.append("| bucket | count | fraction | threshold (vs prompt_len) |")
        lines.append("|---|---|---|---|")
        for b in BUCKET_ORDER:
            c = analysis.bucket_counts.get(b, 0)
            frac = c / total if total > 0 else 0.0
            if b == BUCKET_RECENT:
                thr = f"≤{int(RECENT_FRACTION * 100)}% (or ≤{RECENT_FLOOR})"
            elif b == BUCKET_MEDIUM:
                thr = f"{int(RECENT_FRACTION * 100)}%–{int(MEDIUM_FRACTION * 100)}%"
            elif b == BUCKET_FAR:
                thr = f"{int(MEDIUM_FRACTION * 100)}%–{int(FAR_FRACTION * 100)}%"
            elif b == BUCKET_VERY_FAR:
                thr = f">{int(FAR_FRACTION * 100)}% (start of prompt)"
            else:
                thr = "future (retrieved_pos ≥ query_pos)"
            lines.append(f"| {b} | {c} | {frac:.3f} | {thr} |")
        lines.append("")

    # Sentinel overlap
    if analysis.sentinel_range is not None:
        s_start, s_end = analysis.sentinel_range
        frac = (
            analysis.sentinel_hits / analysis.sentinel_total
            if analysis.sentinel_total > 0
            else 0.0
        )
        lines.append("### Sentinel section retrieval")
        lines.append("")
        lines.append(
            f"- Sentinel position range (tokenized): `{s_start}..{s_end}` (inclusive)"
        )
        lines.append(
            f"- Sentinel hits: **{analysis.sentinel_hits}** / {analysis.sentinel_total} "
            f"(query, layer) pairs"
        )
        lines.append(
            f"- Hit rate: **{frac:.3f}** (fraction of retrieval analyses whose "
            f"top-N included ≥1 sentinel token position)"
        )
        lines.append("")

    # Sample of latest retrieval results
    n_show = min(top_results, len(analysis.results))
    if n_show > 0:
        lines.append(f"### Sample retrieval results (latest {n_show})")
        lines.append("")
        lines.append("| run_id | layer | query_token | top retrieved positions (pos@score, shared) |")
        lines.append("|---|---|---|---|")
        for r in analysis.results[-n_show:]:
            positions_str = ", ".join(
                f"{rp.position}@{rp.score:.3f}({rp.shared_channels})"
                for rp in r.top_positions[:5]
            )
            lines.append(
                f"| `{r.run_id[:10]}` | {r.layer} | {r.query_token} | {positions_str or '_(none)_'} |"
            )
        lines.append("")

    lines.append(
        "_Approximation only — full activation vectors are not stored; this "
        "captures top-K channels per (token, layer). Signed overlap on top-K is "
        "a defensible lower-bound interpretability signal of which earlier "
        "positions share dominant latent dimensions with the current query._ "
        "_Original DSA indexer premise rejected on 2026-06-20 — see "
        "`GLM52_SESSION_MEMORY.md` for the full forensic record._"
    )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_Q_STEM",
    "DEFAULT_K_STEM",
    "DEFAULT_RETRIEVAL_TOPN",
    "BUCKET_ORDER",
    "BUCKET_RECENT",
    "BUCKET_MEDIUM",
    "BUCKET_FAR",
    "BUCKET_VERY_FAR",
    "BUCKET_FUTURE",
    "RetrievedPosition",
    "RetrievalResult",
    "RetrievalAnalysis",
    "signed_overlap",
    "distance_bucket",
    "analyze_retrieval",
    "to_summary_dict",
    "render_markdown",
]
