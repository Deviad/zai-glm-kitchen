"""Aggregate MoE routing traces into the Phase 1 report.

Consumes one or more trace JSONL files (plus their ``.meta.json`` sidecars when
present) and produces:

* a markdown report (``reports/glm52_moe_trace_report.md``) with top experts by
  task / language / layer, overlap scores, router entropy, and prefill-vs-gen
  comparison;
* a machine-readable summary JSON sidecar.

The analyzer is backend-agnostic: it only reads the canonical
:data:`~glm52_kitchen.tracing.schema.EVENT_MOE_ROUTING` records, so it works with the
real C++ ``trace-moe`` backend and with the synthetic generator used in tests.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .schema import (
    EVENT_MOE_ROUTING,
    PHASE_GENERATION,
    PHASE_PREFILL,
    RunMetadata,
    MoeRoutingRecord,
    ActivationSummaryRecord,
    iter_records,
)
from .retrieval import (
    DEFAULT_K_STEM as RETRIEVAL_DEFAULT_K_STEM,
    DEFAULT_RETRIEVAL_TOPN,
    analyze_retrieval,
    render_markdown as render_retrieval_markdown,
    to_summary_dict as retrieval_to_summary_dict,
)

DEFAULT_TOPN = 8


@dataclass
class TraceSource:
    trace_path: str
    meta_path: str | None
    meta: RunMetadata | None
    n_records: int = 0


def load_meta_sidecar(trace_path: str) -> RunMetadata | None:
    p = Path(trace_path)
    meta_path = p.with_suffix(p.suffix + ".meta.json")
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        return RunMetadata.from_dict(obj)
    except Exception as e:
        # Surface schema drift instead of silently dropping metadata: a missing
        # required field here previously made every sidecar fail to load,
        # silently zeroing tokenization stats before the bug was found.
        import sys
        print(f"warn: load_meta_sidecar({p.name}) failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def resolve_traces(patterns: Iterable[str]) -> list[TraceSource]:
    """Expand glob patterns into :class:`TraceSource` objects (one per jsonl)."""
    sources: list[TraceSource] = []
    seen: set[str] = set()
    for pat in patterns:
        # Treat a literal existing file path as a single source even without '*'.
        if os.path.exists(pat) and not any(c in pat for c in "*?["):
            matches = [pat]
        else:
            matches = sorted(glob.glob(pat, recursive=True))
        for path in matches:
            ap = os.path.abspath(path)
            if ap in seen or not path.endswith(".jsonl"):
                continue
            seen.add(ap)
            meta = load_meta_sidecar(path)
            sources.append(
                TraceSource(
                    trace_path=ap,
                    meta_path=str(Path(path).with_suffix(Path(path).suffix + ".meta.json"))
                    if meta
                    else None,
                    meta=meta,
                )
            )
    return sources


# --------------------------------------------------------------------------- aggregation


def _entropy(weights: list[float] | None) -> float | None:
    if not weights:
        return None
    total = sum(weights)
    if total <= 0:
        return 0.0
    h = 0.0
    for w in weights:
        if w <= 0:
            continue
        p = w / total
        h -= p * math.log2(p)
    return h


@dataclass
class Aggregated:
    records: list[MoeRoutingRecord] = field(default_factory=list)
    # Story 6 AC: activation summaries stored separately from routing — different
    # schema, different analysis. Keyed by (task, layer, tensor_stem) for the
    # per-layer-per-tensor stats; top_k_channels pooled into the same Counter.
    activation_summaries: list[ActivationSummaryRecord] = field(default_factory=list)
    # (task, layer) -> Counter[expert]
    by_task_layer: dict[tuple[str, int], Counter] = field(default_factory=lambda: defaultdict(Counter))
    by_lang_layer: dict[tuple[str, int], Counter] = field(default_factory=lambda: defaultdict(Counter))
    by_task_phase_layer: dict[tuple[str, str, int], Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    entropy_by_task: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    entropy_by_lang: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    entropy_by_layer: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    tokens_by_lang: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    layers_seen: set[int] = field(default_factory=set)
    n_expert_used_max: int = 0
    n_expert_total: int | None = None
    sources: list[TraceSource] = field(default_factory=list)


def aggregate(sources: list[TraceSource]) -> Aggregated:
    agg = Aggregated(sources=sources)
    for src in sources:
        for rec in iter_records(src.trace_path):
            src.n_records += 1
            # Story 6 AC: dispatch by record type. ActivationSummaryRecord
            # bypasses the routing-specific aggregation path; it's stored
            # raw in agg.activation_summaries for the activation report.
            if isinstance(rec, ActivationSummaryRecord):
                agg.activation_summaries.append(rec)
                agg.layers_seen.add(rec.layer)
                agg.tokens_by_lang[rec.language] += 1
                continue
            agg.records.append(rec)
            agg.layers_seen.add(rec.layer)
            if rec.n_expert_used is not None:
                agg.n_expert_used_max = max(agg.n_expert_used_max, rec.n_expert_used)
            if rec.n_expert is not None and agg.n_expert_total is None:
                agg.n_expert_total = rec.n_expert
            for e in rec.experts:
                agg.by_task_layer[(rec.task_label, rec.layer)][e] += 1
                agg.by_lang_layer[(rec.language, rec.layer)][e] += 1
                agg.by_task_phase_layer[(rec.task_label, rec.phase, rec.layer)][e] += 1
            if rec.router_entropy is not None:
                agg.entropy_by_task[rec.task_label].append(rec.router_entropy)
                agg.entropy_by_lang[rec.language].append(rec.router_entropy)
                agg.entropy_by_layer[rec.layer].append(rec.router_entropy)
            agg.tokens_by_lang[rec.language] += 1
    return agg


def _top(counter: Counter, n: int) -> list[tuple[int, int]]:
    return counter.most_common(n)


def _jaccard(a: Counter, b: Counter, n: int) -> float:
    sa = {e for e, _ in _top(a, n)}
    sb = {e for e, _ in _top(b, n)}
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


# --------------------------------------------------------------------------- report render


def build_summary(
    agg: Aggregated,
    topn: int = DEFAULT_TOPN,
    *,
    retrieval_q_stem: str | None = None,
    retrieval_k_stem: str | None = None,
    retrieval_topn: int = DEFAULT_RETRIEVAL_TOPN,
    sentinel_position_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    layers = sorted(agg.layers_seen)
    tasks = sorted({t for t, _ in agg.by_task_layer})
    langs = sorted({lang for lang, _ in agg.by_lang_layer})

    top_by_task_layer: dict[str, dict[str, list[list]]] = {}
    for task in tasks:
        per_layer: dict[str, list[list]] = {}
        for layer in layers:
            c = agg.by_task_layer.get((task, layer))
            if c:
                per_layer[str(layer)] = [[int(e), int(n)] for e, n in _top(c, topn)]
        top_by_task_layer[task] = per_layer

    top_by_lang_layer: dict[str, dict[str, list[list]]] = {}
    for lang in langs:
        per_layer = {}
        for layer in layers:
            c = agg.by_lang_layer.get((lang, layer))
            if c:
                per_layer[str(layer)] = [[int(e), int(n)] for e, n in _top(c, topn)]
        top_by_lang_layer[lang] = per_layer

    # task-vs-task overlap (pooled across layers)
    pooled_task = {t: Counter() for t in tasks}
    for (t, _layer), c in agg.by_task_layer.items():
        pooled_task[t] += c
    task_overlap = {}
    for i, a in enumerate(tasks):
        for b in tasks[i + 1 :]:
            task_overlap[f"{a}|{b}"] = round(_jaccard(pooled_task[a], pooled_task[b], topn), 4)

    lang_overlap = {}
    pooled_lang = {lang: Counter() for lang in langs}
    for (lang, _layer), c in agg.by_lang_layer.items():
        pooled_lang[lang] += c
    for i, a in enumerate(langs):
        for b in langs[i + 1 :]:
            lang_overlap[f"{a}|{b}"] = round(_jaccard(pooled_lang[a], pooled_lang[b], topn), 4)

    entropy_summary = {
        "by_task": {t: round(_mean(v), 4) for t, v in agg.entropy_by_task.items()},
        "by_language": {lang: round(_mean(v), 4) for lang, v in agg.entropy_by_lang.items()},
        "by_layer": {str(lang): round(_mean(v), 4) for lang, v in agg.entropy_by_layer.items()},
    }

    # expert specialization: fraction of a task's own top-N experts that are NOT
    # in any other task's top-N (per layer pooled).
    specialization = {}
    for t in tasks:
        mine = {e for e, _ in _top(pooled_task[t], topn)}
        others = set()
        for t2 in tasks:
            if t2 == t:
                continue
            others |= {e for e, _ in _top(pooled_task[t2], topn)}
        if mine:
            specialization[t] = {
                "n_unique_top": len(mine - others),
                "fraction_unique": round(len(mine - others) / len(mine), 4),
            }

    # prefill vs generation (pooled across tasks/layers within each phase)
    prefill_pool: Counter = Counter()
    gen_pool: Counter = Counter()
    for (task, phase, _layer), c in agg.by_task_phase_layer.items():
        if phase == PHASE_PREFILL:
            prefill_pool += c
        elif phase == PHASE_GENERATION:
            gen_pool += c
    phase_top = {
        "prefill": [[int(e), int(n)] for e, n in _top(prefill_pool, topn)],
        "generation": [[int(e), int(n)] for e, n in _top(gen_pool, topn)],
        "jaccard_topn": round(_jaccard(prefill_pool, gen_pool, topn), 4),
    }

    # tokenization stats per language (only if metadata carried prompt_token_count)
    tok_stats: dict[str, dict[str, float]] = {}
    per_lang_tokens: dict[str, list[int]] = defaultdict(list)
    for src in agg.sources:
        if src.meta and src.meta.prompt_token_count:
            per_lang_tokens[src.meta.language].append(src.meta.prompt_token_count)
    for lang, toks in per_lang_tokens.items():
        tok_stats[lang] = {
            "n_runs": len(toks),
            "mean_prompt_tokens": round(_mean([float(x) for x in toks]), 1),
        }

    # Story 6 AC: activation summary stats. Per (task, layer, tensor_stem):
    # mean L2 norm, mean magnitude, std, max_abs, plus the channel-freq
    # Counter pooled across tokens. Activation records may be empty if the
    # C++ tracer was run without --trace-activations — skip section entirely.
    activation_summary: dict[str, Any] = {}
    if agg.activation_summaries:
        # Per (task, layer, tensor_stem) pooled channel-frequency counter.
        # top_k_channels arrives as [[channel_idx, magnitude], ...] per token;
        # we count how often each channel appears in any token's top-K.
        ch_freq: dict[tuple[str, int, str], Counter] = defaultdict(Counter)
        # Story 6+: parallel by-language grouping. Cheap (one extra Counter
        # per (language, layer, tensor_stem) key) and lets the cross-task
        # analyzer also compute cross-language Jaccard on a single trace.
        ch_freq_by_lang: dict[tuple[str, int, str], Counter] = defaultdict(Counter)
        per_group_norm: dict[tuple[str, int, str], list[float]] = defaultdict(list)
        per_group_mean: dict[tuple[str, int, str], list[float]] = defaultdict(list)
        per_group_std: dict[tuple[str, int, str], list[float]] = defaultdict(list)
        per_group_max: dict[tuple[str, int, str], list[float]] = defaultdict(list)
        per_group_topk: dict[tuple[str, int, str], int] = {}
        per_group_n_channels: dict[tuple[str, int, str], int] = {}
        for rec in agg.activation_summaries:
            key = (rec.task_label, rec.layer, rec.tensor_stem)
            for channel_idx, _mag in rec.top_k_channels:
                ch_freq[key][int(channel_idx)] += 1
            if rec.language:
                lang_key = (rec.language, rec.layer, rec.tensor_stem)
                for channel_idx, _mag in rec.top_k_channels:
                    ch_freq_by_lang[lang_key][int(channel_idx)] += 1
            if rec.l2_norm is not None:
                per_group_norm[key].append(rec.l2_norm)
            if rec.mean is not None:
                per_group_mean[key].append(rec.mean)
            if rec.std is not None:
                per_group_std[key].append(rec.std)
            if rec.max_abs is not None:
                per_group_max[key].append(rec.max_abs)
            if rec.topk is not None:
                per_group_topk[key] = rec.topk
            if rec.n_channels is not None:
                per_group_n_channels[key] = rec.n_channels
        # Render per (task, layer, tensor_stem) summary row
        rows = []
        for key, freq in ch_freq.items():
            task, layer, stem = key
            row = {
                "task": task,
                "layer": layer,
                "tensor_stem": stem,
                "n_tokens": sum(freq.values()),  # total (token, channel) pairs
                "topk": per_group_topk.get(key),
                "n_channels": per_group_n_channels.get(key),
                "mean_l2_norm": round(_mean(per_group_norm[key]), 4) if per_group_norm[key] else None,
                "mean_activation_mean": round(_mean(per_group_mean[key]), 4) if per_group_mean[key] else None,
                "mean_activation_std": round(_mean(per_group_std[key]), 4) if per_group_std[key] else None,
                "mean_max_abs": round(_mean(per_group_max[key]), 4) if per_group_max[key] else None,
                "top_channels_by_frequency": _top(freq, topn),
            }
            rows.append(row)
        # Sort: by task, then tensor_stem, then layer — stable for diff-friendly output.
        rows.sort(key=lambda r: (r["task"], r["tensor_stem"], r["layer"]))
        activation_summary["n_activation_records"] = len(agg.activation_summaries)
        activation_summary["rows"] = rows
        # Parallel by-language rows (Story 6+). Symmetric to `rows` but keyed
        # by language label. Smaller numbers (per-language aggregation pool)
        # but enough for cross-language Jaccard. Empty when records have no
        # language set (e.g. synth traces without language args).
        lang_rows = []
        if ch_freq_by_lang:
            for key, freq in ch_freq_by_lang.items():
                language, layer, stem = key
                lang_rows.append({
                    "language": language,
                    "layer": layer,
                    "tensor_stem": stem,
                    "n_tokens": sum(freq.values()),
                    "top_channels_by_frequency": _top(freq, topn),
                })
            lang_rows.sort(key=lambda r: (r["language"], r["tensor_stem"], r["layer"]))
        activation_summary["rows_by_language"] = lang_rows

    # Story 5 (re-scoped 2026-06-20): MLA retrieval-pattern analysis over the
    # activation summaries. Only runs when caller asks for it by setting
    # retrieval_q_stem (typically "q_nope_absorbed"). Pulls both query+key
    # records from the existing agg.activation_summaries list.
    retrieval_summary: dict[str, Any] = {}
    if retrieval_q_stem and agg.activation_summaries:
        q_stem = retrieval_q_stem
        k_stem = retrieval_k_stem or RETRIEVAL_DEFAULT_K_STEM
        analysis = analyze_retrieval(
            agg.activation_summaries,
            q_stem=q_stem,
            k_stem=k_stem,
            topn=retrieval_topn,
            sentinel_range=sentinel_position_range,
        )
        retrieval_summary = retrieval_to_summary_dict(analysis)
        retrieval_summary["markdown_section"] = render_retrieval_markdown(analysis)

    return {
        "schema_version": 1,
        "n_records": len(agg.records),
        "n_sources": len(agg.sources),
        "layers": layers,
        "tasks": tasks,
        "languages": langs,
        "n_expert_used_max": agg.n_expert_used_max,
        "n_expert_total": agg.n_expert_total,
        "topn": topn,
        "top_experts_by_task_layer": top_by_task_layer,
        "top_experts_by_language_layer": top_by_lang_layer,
        "task_overlap_jaccard_topn": task_overlap,
        "language_overlap_jaccard_topn": lang_overlap,
        "router_entropy": entropy_summary,
        "expert_specialization": specialization,
        "prefill_vs_generation": phase_top,
        "tokenization_stats_per_language": tok_stats,
        "tokens_traced_per_language": {lang: agg.tokens_by_lang[lang] for lang in langs},
        # Story 6 AC: bounded activation summary section. Absent when the trace
        # was produced without --trace-activations (no activation records).
        "activation_summary": activation_summary,
        # Story 5 (re-scoped): MLA retrieval-pattern analysis. Populated only
        # when caller set retrieval_q_stem AND activation_summaries are present.
        "retrieval_analysis": retrieval_summary,
        "sources": [
            {
                "trace": os.path.basename(s.trace_path),
                "records": s.n_records,
                "run_id": s.meta.run_id if s.meta else None,
                "model": s.meta.model if s.meta else None,
                "language": s.meta.language if s.meta else None,
                "task_label": s.meta.task_label if s.meta else None,
                "perf_gen_per_sec": s.meta.perf_gen_per_sec if s.meta else None,
                "perf_prompt_eval_per_sec": s.meta.perf_prompt_eval_per_sec if s.meta else None,
                "perf_prompt_eval_ms": s.meta.perf_prompt_eval_ms if s.meta else None,
                "perf_eval_ms": s.meta.perf_eval_ms if s.meta else None,
                "perf_n_prompt_eval": s.meta.perf_n_prompt_eval if s.meta else None,
                "perf_n_eval": s.meta.perf_n_eval if s.meta else None,
                "n_expert_total": (s.meta.n_expert_total if s.meta and s.meta.n_expert_total else None)
                    or (agg.n_expert_total if agg.n_expert_total else None),
                "backpressure": s.meta.backpressure if s.meta else None,
                "records_dropped": s.meta.records_dropped if s.meta else None,
                "records_sampled": s.meta.records_sampled if s.meta else None,
                # Story 9 AC: real provenance fields (no more placeholders).
                "command_line": s.meta.command_line if s.meta else None,
                "prompt_sha256": s.meta.prompt_sha256 if s.meta else None,
                "model_sha256_prefix": s.meta.model_sha256_prefix if s.meta else None,
                "model_size_bytes": s.meta.model_size_bytes if s.meta else None,
                "model_total_size_bytes": s.meta.model_total_size_bytes if s.meta else None,
                "started_at": s.meta.started_at if s.meta else None,
                "ended_at": s.meta.ended_at if s.meta else None,
            }
            for s in agg.sources
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GLM-5.2 MoE Expert-Routing Trace Report")
    lines.append("")
    # Story 9 AC: reproducibility provenance. Pull from the first source with
    # populated meta — all runs in a batched study share the same model, so one
    # representative is enough to reproduce. Skip section if no sidecars exist.
    srcs = summary.get("sources") or []
    prov = next((s for s in srcs if s.get("model")), None)
    if prov and prov.get("model"):
        lines.append("## Reproducibility provenance")
        lines.append("")
        lines.append(
            "All runs in this report traced the same model. Re-run with:"
        )
        lines.append("")
        lines.append("```text")
        # command_line may be missing or placeholder for old sidecars.
        cl = prov.get("command_line") or "(unavailable)"
        lines.append(cl)
        lines.append("```")
        lines.append("")
        lines.append(f"- Model: `{prov['model']}`")
        if prov.get("prompt_sha256") and prov["prompt_sha256"] != "(see run log)":
            lines.append(f"- prompt_sha256 (this run): `{prov['prompt_sha256']}`")
        if prov.get("model_sha256_prefix"):
            lines.append(f"- model_sha256_prefix (first 1 MiB): `{prov['model_sha256_prefix']}`")
        if prov.get("model_total_size_bytes"):
            g = prov["model_total_size_bytes"] / (1024**3)
            lines.append(f"- model_total_size_bytes: **{prov['model_total_size_bytes']}** ({g:.1f} GiB)")
        elif prov.get("model_size_bytes"):
            # Field exists but only single-shard size — flag so reader knows
            # total may differ for multi-shard models.
            lines.append(f"- model_size_bytes (single shard): `{prov['model_size_bytes']}`")
        if prov.get("started_at") and prov.get("ended_at"):
            lines.append(f"- Run window: `{prov['started_at']}` → `{prov['ended_at']}` (UTC)")
        # Story 8 AC: speed metrics per source (only when populated)
        if prov.get("perf_gen_per_sec"):
            lines.append(
                f"- Speed: **{prov['perf_gen_per_sec']:.2f} gen tok/s**, "
                f"{prov.get('perf_prompt_eval_per_sec') or 0:.2f} prefill tok/s "
                f"({prov.get('perf_n_eval') or 0} gen tokens / {prov.get('perf_n_prompt_eval') or 0} prompt tokens)"
            )
        # Story 8 AC: n_expert_total per the GGUF KV (256 for GLM-5.2).
        if prov.get("n_expert_total"):
            lines.append(f"- n_expert_total: **{prov['n_expert_total']}** (total routed experts per MoE layer)")
        if prov.get("run_id"):
            lines.append(f"- Sample run_id: `{prov['run_id']}`")
        lines.append("")
    lines.append(
        f"- Records traced: **{summary['n_records']}** across **{summary['n_sources']}** run(s)"
    )
    lines.append(f"- Layers seen: **{len(summary['layers'])}** ({summary['layers'][0]}..{summary['layers'][-1]})"
                 if summary["layers"] else "- Layers seen: 0")
    if summary.get("n_expert_total"):
        lines.append(
            f"- Experts per routing event: up to **{summary['n_expert_used_max']}** "
            f"of **{summary['n_expert_total']}** total"
        )
    else:
        lines.append(f"- Experts per routing event: up to **{summary['n_expert_used_max']}**")
    lines.append(f"- Tasks: `{', '.join(summary['tasks']) or 'n/a'}`")
    lines.append(f"- Languages: `{', '.join(summary['languages']) or 'n/a'}`")
    lines.append("")

    def _fmt_top(rows: list[list[int]]) -> str:
        if not rows:
            return "_(none)_"
        return ", ".join(f"#{e}×{n}" for e, n in rows)

    lines.append("## Top experts by task and layer")
    for task, per_layer in summary["top_experts_by_task_layer"].items():
        lines.append(f"### Task: `{task}`")
        lines.append("| layer | top experts (id×count) |")
        lines.append("|---|---|")
        for layer in sorted(map(int, per_layer.keys())):
            rows = per_layer[str(layer)]
            lines.append(f"| {layer} | {_fmt_top(rows)} |")
        lines.append("")

    lines.append("## Top experts by language and layer")
    for lang, per_layer in summary["top_experts_by_language_layer"].items():
        lines.append(f"### Language: `{lang}`")
        # Show only a few representative layers to keep the report readable.
        all_layers = sorted(map(int, per_layer.keys()))
        show = all_layers
        if len(all_layers) > 6:
            mid = all_layers[len(all_layers) // 2]
            show = [all_layers[0], mid, all_layers[-1]]
        lines.append("| layer | top experts (id×count) |")
        lines.append("|---|---|")
        for layer in show:
            rows = per_layer[str(layer)]
            lines.append(f"| {layer} | {_fmt_top(rows)} |")
        lines.append("")

    lines.append("## Router entropy")
    e = summary["router_entropy"]
    lines.append("| group | mean entropy (bits) |")
    lines.append("|---|---|")
    for t, v in sorted(e["by_task"].items()):
        lines.append(f"| task `{t}` | {v} |")
    for lang, v in sorted(e["by_language"].items()):
        lines.append(f"| lang `{lang}` | {v} |")
    lines.append("")

    lines.append("## Task overlap (Jaccard of pooled top-N experts)")
    lines.append("| task pair | jaccard |")
    lines.append("|---|---|")
    for pair, v in sorted(summary["task_overlap_jaccard_topn"].items()):
        lines.append(f"| {pair} | {v} |")
    lines.append("")
    lines.append("## Language overlap (Jaccard of pooled top-N experts)")
    lines.append("| language pair | jaccard |")
    lines.append("|---|---|")
    for pair, v in sorted(summary["language_overlap_jaccard_topn"].items()):
        lines.append(f"| {pair} | {v} |")
    lines.append("")

    lines.append("## Expert specialization (fraction of a task's top-N unique to it)")
    lines.append("| task | unique top-N | fraction unique |")
    lines.append("|---|---|---|")
    for t, d in sorted(summary["expert_specialization"].items()):
        lines.append(f"| {t} | {d['n_unique_top']} | {d['fraction_unique']} |")
    lines.append("")

    pv = summary["prefill_vs_generation"]
    lines.append("## Prefill vs generation")
    lines.append("| phase | top experts |")
    lines.append("|---|---|")
    lines.append(f"| prefill | {_fmt_top(pv['prefill'])} |")
    lines.append(f"| generation | {_fmt_top(pv['generation'])} |")
    lines.append(f"\nJaccard (top-N) between phases: **{pv['jaccard_topn']}**")
    lines.append("")

    tok = summary.get("tokenization_stats_per_language") or {}
    if tok:
        lines.append("## Tokenization stats per language (from metadata)")
        lines.append("| language | runs | mean prompt tokens |")
        lines.append("|---|---|---|---|")
        for lang, d in sorted(tok.items()):
            lines.append(f"| {lang} | {d['n_runs']} | {d['mean_prompt_tokens']} |")
        lines.append("")

    # Story 6 AC: bounded activation summary section. Rendered only if the
    # trace was produced with --trace-activations; absent means absent.
    act = summary.get("activation_summary") or {}
    if act and act.get("rows"):
        lines.append("## Bounded activation summaries (Phase 4)")
        lines.append("")
        lines.append(
            f"- Activation summary records: **{act['n_activation_records']}** "
            f"across **{len(act['rows'])}** (task, layer, tensor) groups"
        )
        lines.append("")
        lines.append("| task | layer | tensor_stem | topk | n_channels | n_tokens | mean L2 | mean mean | mean std | mean max_abs | top channels |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in act["rows"]:
            top_ch = ", ".join(
                f"#{c}" for c, _ in r["top_channels_by_frequency"]
            )
            lines.append(
                f"| {r['task']} | {r['layer']} | `{r['tensor_stem']}` | "
                f"{r['topk']} | {r['n_channels']} | {r['n_tokens']} | "
                f"{r['mean_l2_norm']} | {r['mean_activation_mean']} | "
                f"{r['mean_activation_std']} | {r['mean_max_abs']} | {top_ch} |"
            )
        lines.append("")
        lines.append(
            "_Top channels ranked by frequency-of-appearance in any token's"
            " top-K activation list (not by magnitude). n_tokens is the total"
            " (token, channel) pair count contributing to that group. Stat"
            " columns are means across per-token values._"
        )
        lines.append("")

    # Story 5 (re-scoped): MLA retrieval-pattern analysis. The markdown is
    # pre-rendered in build_summary so we just splice it in.
    retr = summary.get("retrieval_analysis") or {}
    retr_md = retr.get("markdown_section")
    if retr_md:
        lines.append(retr_md)

    lines.append("## Runs")
    lines.append("| trace | records | language | task | gen/s | dropped | sampled |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in summary["sources"]:
        lines.append(
            f"| {s['trace']} | {s['records']} | {s['language']} | {s['task_label']} | "
            f"{s['perf_gen_per_sec']} | {s['records_dropped']} | {s['records_sampled']} |"
        )
    lines.append("")
    if retr_md:
        lines.append(
            "MLA retrieval-pattern tracing (Phase 3 / Story 5 re-scoped) is **on** — the report above "
            "includes the MLA retrieval analysis section. Activation summaries (Phase 4) are also "
            "present. See `GLM52_TRACE_PLAN.md` for the re-scope rationale."
        )
    else:
        lines.append(
            "MLA retrieval-pattern tracing (Phase 3 / Story 5 re-scoped) is **not enabled** in "
            "this report — re-run the analyzer with `--retrieval-stems <q>,<k>` and traces "
            "captured via `--trace-activations <q>,<k>`. Activation summaries (Phase 4) are "
            "disabled by default and require explicit flags."
        )
    md = "\n".join(lines) + "\n"
    # Stable content hash so report provenance can reference the exact summary.
    return md


def write_report(
    sources: list[TraceSource],
    out_md: str | os.PathLike[str],
    out_json: str | os.PathLike[str],
    *,
    topn: int = DEFAULT_TOPN,
    retrieval_q_stem: str | None = None,
    retrieval_k_stem: str | None = None,
    retrieval_topn: int = DEFAULT_RETRIEVAL_TOPN,
    sentinel_position_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    agg = aggregate(sources)
    summary = build_summary(
        agg,
        topn=topn,
        retrieval_q_stem=retrieval_q_stem,
        retrieval_k_stem=retrieval_k_stem,
        retrieval_topn=retrieval_topn,
        sentinel_position_range=sentinel_position_range,
    )
    summary["report_sha256"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True).encode("utf-8")
    ).hexdigest()
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    md = render_markdown(summary)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return summary


__all__ = [
    "Aggregated",
    "TraceSource",
    "aggregate",
    "build_summary",
    "iter_records",
    "load_meta_sidecar",
    "render_markdown",
    "resolve_traces",
    "write_report",
    "DEFAULT_TOPN",
    "EVENT_MOE_ROUTING",
    "DEFAULT_RETRIEVAL_TOPN",
    "PHASE_PREFILL",
    "PHASE_GENERATION",
]
