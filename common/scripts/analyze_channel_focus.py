#!/usr/bin/env python3
"""Single-channel focus analyzer (Phase 6 — investigate channel #4386).

The Phase 5 + 5b studies established that channel #4386 is a task-agnostic
backbone — it's in the shared core at every layer 0..72 in both pilot
(N=49) and full (N=161) runs. This script answers the natural follow-up:
**what makes #4386 fire?**

Reads raw .jsonl trace files (the per-token activation_summary records)
and aggregates, for a single channel index:

  - Mean / std magnitude by (task, layer, phase)
  - Mean / std magnitude by (language, layer, phase)
  - Rank distribution: where does 4386 sit in the per-token top-K?
  - High-firing tokens: token_index positions with the highest 4386
    magnitude, broken down by phase (prefill vs generation)
  - Complements: which OTHER channels tend to fire alongside #4386
    (computed over tokens where #4386 is rank-1)

The Phase 6 follow-up question: is #4386 a positional primitive (fires on
the FIRST tokens, regardless of content), a structural marker (punctuation,
BOS, etc.), or a genuine semantic primitive?
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


DEFAULT_CHANNEL = 4386  # robust Phase 5 finding: universal task-agnostic channel


def iter_trace_files(patterns: list[str]) -> Iterable[str]:
    seen: set[str] = set()
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            if f.endswith(".meta.json"):
                continue
            rp = os.path.realpath(f)
            if rp in seen:
                continue
            seen.add(rp)
            yield f


def load_meta_sidecar(trace_path: str) -> dict:
    sidecar = trace_path + ".meta.json"
    if os.path.exists(sidecar):
        try:
            return json.loads(Path(sidecar).read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def iter_activation_records(trace_path: str) -> Iterable[dict]:
    with open(trace_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") != "activation_summary":
                continue
            yield r


def find_channel_in_topk(
    top_k_channels: list, target: int
) -> tuple[int, float] | None:
    """Return (rank_1_indexed, magnitude) if `target` is in the top-K list.

    `top_k_channels` arrives as [[channel_idx, magnitude], ...] sorted
    descending by absolute magnitude (per-token). Rank 1 = highest-magnitude.
    """
    for i, pair in enumerate(top_k_channels):
        if len(pair) >= 1 and int(pair[0]) == target:
            mag = float(pair[1]) if len(pair) >= 2 else 0.0
            return (i + 1, mag)
    return None


def analyze(
    trace_paths: list[str],
    channel: int,
    top_fraction: float = 0.10,
) -> dict:
    """Aggregate #channel-stats across all trace paths.

    Returns a JSON-serializable dict with the breakdowns.
    """
    # Per (task, layer, phase) magnitude lists.
    by_task_layer_phase: dict[tuple, list[float]] = defaultdict(list)
    by_lang_layer_phase: dict[tuple, list[float]] = defaultdict(list)
    # Rank distribution (overall): rank -> count
    rank_overall: Counter = Counter()
    rank_by_task_phase: dict[tuple, Counter] = defaultdict(Counter)
    # Token-index positions where #channel is highest-magnitude (rank-1)
    # Bucketed by phase and normalized as fraction of prompt_len where possible.
    rank1_positions_by_phase_task: dict[tuple, list[int]] = defaultdict(list)
    rank1_positions_by_phase_lang: dict[tuple, list[int]] = defaultdict(list)
    # Channels that co-fire with target when target is rank-1
    cofire_when_rank1: Counter = Counter()
    # Track prompt_len per record (we'll need it for position normalization).
    # The per-record prompt_len isn't stored on each activation record, but
    # the meta sidecar has prompt_token_count per trace file. We'll attach it.
    per_run_prompt_len: dict[str, int] = {}
    # Bucket tokens where #channel is in top-K: list of (run_id, token_index,
    # layer, rank, magnitude, phase, task, language, prompt_len)
    # Not necessary — we'll aggregate inline.

    total_records = 0
    records_with_channel = 0
    records_rank1 = 0

    for trace_path in trace_paths:
        meta = load_meta_sidecar(trace_path)
        prompt_len = int(meta.get("prompt_token_count") or 0)
        run_id = meta.get("run_id") or ""
        if run_id and prompt_len:
            per_run_prompt_len[run_id] = prompt_len

        for rec in iter_activation_records(trace_path):
            total_records += 1
            top_k = rec.get("top_k_channels") or []
            tuple_pair = find_channel_in_topk(top_k, channel)
            if tuple_pair is None:
                continue
            rank, mag = tuple_pair
            records_with_channel += 1
            task = rec.get("task_label") or "?"
            lang = rec.get("language") or "?"
            phase = rec.get("phase", "?")
            layer = int(rec.get("layer", -1))
            token_index = int(rec.get("token_index", -1))
            run_id = rec.get("run_id", "")

            by_task_layer_phase[(task, layer, phase)].append(mag)
            by_lang_layer_phase[(lang, layer, phase)].append(mag)
            rank_overall[rank] += 1
            rank_by_task_phase[(task, phase)][rank] += 1

            if rank == 1:
                records_rank1 += 1
                rank1_positions_by_phase_task[(phase, task)].append(token_index)
                rank1_positions_by_phase_lang[(phase, lang)].append(token_index)
                # Co-firing channels
                for pair in top_k:
                    if len(pair) >= 1 and int(pair[0]) != channel:
                        cofire_when_rank1[int(pair[0])] += 1

    # Bucket positions to fractions of prompt_len when prompt_len is known.
    # For prefill, token_index in [0, prompt_len-1].
    # For generation, token_index >= prompt_len. We'll normalize separately.
    def bucket_positions(
        positions: list[int], prompt_lens: list[int], phase: str
    ) -> dict:
        """Return position distribution (fraction buckets) for the given
        positions. For prefill: frac = token_index / max(prompt_len, 1).
        For generation: token_index retraces as offset from prompt_len."""
        if not positions:
            return {}
        buckets = Counter()
        for i, pos in enumerate(positions):
            pl = prompt_lens[i] if i < len(prompt_lens) else 0
            if phase == "prefill" and pl > 0:
                frac = pos / pl
            elif phase == "generation" and pl > 0:
                frac = (pos - pl) / max(1, pl)  # normalized gen offset
            else:
                frac = None
            if frac is None:
                buckets["unknown"] += 1
                continue
            if frac < 0:
                buckets["<0 (gen before start)"] += 1
            elif frac < 0.1:
                buckets["0-10%"] += 1
            elif frac < 0.25:
                buckets["10-25%"] += 1
            elif frac < 0.5:
                buckets["25-50%"] += 1
            elif frac < 0.75:
                buckets["50-75%"] += 1
            elif frac < 0.9:
                buckets["75-90%"] += 1
            else:
                buckets["90-100%"] += 1
        return dict(buckets)

    # Build the per-task-phase position buckets. Need per-position prompt_len
    # — but we tracked per-run, and rank1_positions rely on per-run_id prompt_len.
    # We don't keep run_id in the lists above (we collapsed to (phase, task)).
    # Workaround: re-iterate to compute buckets... but that's expensive.
    # Simpler: for position buckets, re-iterate just to count fraction-buckets.
    # We'll do this as a separate pass.
    position_buckets_by_phase_task: dict[tuple, dict] = defaultdict(dict)
    # One more pass: for each rank1 record, get the prompt_len from the
    # trace's meta sidecar; compute frac; bucket.
    # Cache per-trace-file prompt_len to avoid re-reading meta.
    per_file_prompt_len: dict[str, int] = {}
    for trace_path in trace_paths:
        if trace_path not in per_file_prompt_len:
            meta = load_meta_sidecar(trace_path)
            per_file_prompt_len[trace_path] = int(meta.get("prompt_token_count") or 0)
        pl = per_file_prompt_len[trace_path]
        for rec in iter_activation_records(trace_path):
            top_k = rec.get("top_k_channels") or []
            tuple_pair = find_channel_in_topk(top_k, channel)
            if tuple_pair is None or tuple_pair[0] != 1:
                continue
            task = rec.get("task_label") or "?"
            lang = rec.get("language") or "?"
            phase = rec.get("phase") or "?"
            tok_idx = int(rec.get("token_index", -1))
            if phase == "prefill" and pl > 0:
                frac = tok_idx / pl
            elif phase == "generation" and pl > 0:
                frac = (tok_idx - pl) / max(1, pl)
            else:
                continue
            b_key = None
            if frac < 0:
                b_key = "<0"
            elif frac < 0.1:
                b_key = "0-10%"
            elif frac < 0.25:
                b_key = "10-25%"
            elif frac < 0.5:
                b_key = "25-50%"
            elif frac < 0.75:
                b_key = "50-75%"
            elif frac < 0.9:
                b_key = "75-90%"
            else:
                b_key = "90-100%"
            curr = position_buckets_by_phase_task[(phase, task)]
            curr[b_key] = curr.get(b_key, 0) + 1

    # Aggregate stats.
    def stats(lst: list[float]) -> dict:
        if not lst:
            return {"n": 0}
        return {
            "n": len(lst),
            "mean": round(mean(lst), 6),
            "std": round(pstdev(lst) if len(lst) > 1 else 0.0, 6),
            "min": round(min(lst), 6),
            "max": round(max(lst), 6),
        }

    # Magnitude breakdown — flatten to per-task-phase summaries (averaging
    # the per-layer means).
    task_phase_summary: dict[tuple, dict] = {}
    for (task, _layer, phase), mags in by_task_layer_phase.items():
        key = (task, phase)
        if key not in task_phase_summary:
            task_phase_summary[key] = {"all": []}
        task_phase_summary[key]["all"].extend(mags)
    task_phase_table = []
    for (task, phase), d in sorted(task_phase_summary.items()):
        s = stats(d["all"])
        s.update({"task": task, "phase": phase})
        task_phase_table.append(s)

    lang_phase_table = []
    lang_phase_summary: dict[tuple, list[float]] = defaultdict(list)
    for (lang, _layer, phase), mags in by_lang_layer_phase.items():
        lang_phase_summary[(lang, phase)].extend(mags)
    for (lang, phase), lst in sorted(lang_phase_summary.items()):
        s = stats(lst)
        s.update({"language": lang, "phase": phase})
        lang_phase_table.append(s)

    # Per-layer mean magnitude (one number per layer, averaging across all
    # tasks/languages/phases). Reveals whether #channel magnitude varies
    # by depth.
    per_layer: dict[int, list[float]] = defaultdict(list)
    for (_task, layer, _phase), mags in by_task_layer_phase.items():
        per_layer[layer].extend(mags)
    per_layer_table = [
        {"layer": layer, **stats(per_layer[layer])}
        for layer in sorted(per_layer)
    ]

    return {
        "channel": channel,
        "n_trace_files": len(trace_paths),
        "n_activation_records_scanned": total_records,
        "n_records_with_channel": records_with_channel,
        "n_records_rank1": records_rank1,
        "channel_appearance_rate": (
            round(records_with_channel / total_records, 4)
            if total_records else 0.0
        ),
        "rank1_rate": (
            round(records_rank1 / total_records, 4)
            if total_records else 0.0
        ),
        "rank_distribution_overall": dict(sorted(rank_overall.items())),
        "task_phase_magnitude": task_phase_table,
        "language_phase_magnitude": lang_phase_table,
        "per_layer_magnitude": per_layer_table,
        "position_buckets_by_phase_task_rank1": {
            f"{phase}|{task}": buckets
            for (phase, task), buckets in sorted(position_buckets_by_phase_task.items())
        },
        "cofire_channels_when_rank1_top20": dict(cofire_when_rank1.most_common(20)),
    }


def render_markdown(s: dict, channel: int) -> str:
    out: list[str] = []
    out.append(f"# Phase 6 — channel #{channel} focus investigation\n")
    out.append(
        f"Scanned {s['n_activation_records_scanned']:,} activation records "
        f"across {s['n_trace_files']} trace files. Channel #{channel} "
        f"appeared in {s['n_records_with_channel']:,} of them "
        f"({s['channel_appearance_rate']*100:.1f}%) and was the "
        f"rank-1 (highest-magnitude) channel in {s['n_records_rank1']:,} "
        f"({s['rank1_rate']*100:.1f}%).\n"
    )

    out.append("\n## Rank distribution (when #channel is in top-K)\n")
    out.append("| rank | count | % |\n|---|---|---|")
    total_with = s["n_records_with_channel"]
    for rank, count in sorted(s["rank_distribution_overall"].items()):
        pct = (count / total_with * 100) if total_with else 0.0
        out.append(f"| {rank} | {count:,} | {pct:.1f}% |")

    out.append("\n## Per-layer magnitude\n")
    out.append("| layer | n | mean | std | min | max |\n|---|---|---|---|---|---|")
    for row in s["per_layer_magnitude"]:
        out.append(
            f"| {row['layer']} | {row['n']:,} | {row['mean']} | "
            f"{row.get('std', 0.0)} | {row.get('min', 0.0)} | {row.get('max', 0.0)} |"
        )

    out.append("\n## Mean magnitude by (task, phase)\n")
    out.append(
        "| task | phase | n | mean | std | min | max |\n"
        "|---|---|---|---|---|---|---|"
    )
    for row in s["task_phase_magnitude"]:
        out.append(
            f"| {row['task']} | {row['phase']} | {row['n']:,} | "
            f"{row['mean']} | {row.get('std', 0.0)} | "
            f"{row.get('min', 0.0)} | {row.get('max', 0.0)} |"
        )

    out.append("\n## Mean magnitude by (language, phase)\n")
    out.append(
        "| language | phase | n | mean | std | min | max |\n"
        "|---|---|---|---|---|---|---|"
    )
    for row in s["language_phase_magnitude"]:
        out.append(
            f"| {row['language']} | {row['phase']} | {row['n']:,} | "
            f"{row['mean']} | {row.get('std', 0.0)} | "
            f"{row.get('min', 0.0)} | {row.get('max', 0.0)} |"
        )

    out.append(
        "\n## Position distribution when #channel is rank-1\n"
        "(token position as fraction of prompt_len — prefill phase)\n"
    )
    out.append("| task | 0-10% | 10-25% | 25-50% | 50-75% | 75-90% | 90-100% | other |")
    out.append("|---|---|---|---|---|---|---|---|")
    pos_data = s["position_buckets_by_phase_task_rank1"]
    tasks = sorted({k.split("|", 1)[1] for k in pos_data if k.startswith("prefill|")})
    for task in tasks:
        b = pos_data.get(f"prefill|{task}", {})
        other = b.get("<0", 0) + b.get("unknown", 0)
        out.append(
            f"| {task} | {b.get('0-10%', 0)} | {b.get('10-25%', 0)} | "
            f"{b.get('25-50%', 0)} | {b.get('50-75%', 0)} | "
            f"{b.get('75-90%', 0)} | {b.get('90-100%', 0)} | {other} |"
        )

    out.append(
        "\n## Co-firing channels (top-20, when #channel is rank-1)\n"
        "These channels appear alongside #" + str(channel) + " in the same token's top-K.\n"
    )
    out.append("| channel | co-occurrences |\n|---|---|")
    for c, n in s["cofire_channels_when_rank1_top20"].items():
        out.append(f"| #{c} | {n:,} |")

    out.append(
        "\n_Approximation: channel magnitudes are only collected when the "
        "target channel is already in that token's top-K (default 15). "
        "Records where the channel ranks lower than top-K are not visible "
        "to this analyzer. See GLM52_SESSION_MEMORY.md for scope._"
    )
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--traces",
        nargs="+",
        required=True,
        help="Trace file paths or globs (e.g. 'traces/batch/activation_*/ *.jsonl')",
    )
    ap.add_argument(
        "--channel",
        type=int,
        default=DEFAULT_CHANNEL,
        help=f"Target channel index to investigate (default: {DEFAULT_CHANNEL})",
    )
    ap.add_argument("--out-md", default=None, help="Output markdown report path")
    ap.add_argument("--out-json", default=None, help="Output JSON summary path")
    args = ap.parse_args()

    trace_paths = list(iter_trace_files(args.traces))
    if not trace_paths:
        print(f"ERROR: no trace files matched: {args.traces}", file=sys.stderr)
        return 2

    print(f"# analyzing channel #{args.channel} across {len(trace_paths)} files",
          file=sys.stderr)
    s = analyze(trace_paths, args.channel)
    md = render_markdown(s, args.channel)

    if args.out_md:
        Path(args.out_md).write_text(md, encoding="utf-8")
        print(f"wrote {args.out_md}", file=sys.stderr)
    else:
        print(md)

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(s, indent=2, default=str), encoding="utf-8"
        )
        print(f"wrote {args.out_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
