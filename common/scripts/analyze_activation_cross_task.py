#!/usr/bin/env python3
"""Cross-task / cross-language activation-channel comparison.

Reads the standard `analyze_moe_trace.py` summary JSON (must contain
`activation_summary.rows`) and computes, per (layer, tensor_stem):

  - Pairwise Jaccard overlap of top-K channels between every task pair
    (lower overlap => more task-specific channel selection)
  - Pairwise Jaccard overlap between every language pair
    (lower overlap => more language-specific channel selection)
  - Per-layer "shared core" = channels appearing in >= K_shared of N tasks
    (the channel sub-population that is task-agnostic)
  - Per-layer "task-specific" = channels appearing in <= K_specific of N tasks
    (the channels where the task is doing something different)
  - Per-task signature = the channels that distinguish it most from other tasks

The intent is to answer the research question: do different TASKS activate
different channels? Do different LANGUAGES? The default analyzer shows
top channels per (task, layer) but doesn't directly compute dissimilarity
across them; this script closes that gap.

Output: markdown report (one `## Cross-task activation` section) + a
JSON summary file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def topn_channels(row: dict, n: int) -> set[int]:
    """Extract top-N channel IDs (by frequency) from an activation row.

    `top_channels_by_frequency` is a list of [channel_idx, count] pairs
    already sorted descending by the analyzer. Slice top-N.
    """
    pairs = row.get("top_channels_by_frequency") or []
    # Defensive: handle either a list-of-pairs or a list-of-dicts
    out: list[int] = []
    for p in pairs[:n]:
        if isinstance(p, list) and len(p) >= 1:
            out.append(int(p[0]))
        elif isinstance(p, dict) and "channel" in p:
            out.append(int(p["channel"]))
    return set(out)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0  # both empty = identical
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def layer_groups(rows: Iterable[dict]) -> dict[tuple[int, str], list[dict]]:
    """Group activation rows by (layer, tensor_stem).

    Each row in the summary is already (task, layer, tensor_stem)-granular.
    We want (layer, tensor_stem) -> list of per-task rows so we can run
    pairwise comparisons.
    """
    by_layer: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (int(r["layer"]), r["tensor_stem"])
        by_layer[key].append(r)
    return by_layer


def per_task_top_channels(
    rows: list[dict], topn: int
) -> dict[str, set[int]]:
    """Map task -> set of top-N channels (across all rows of that task
    that share the same (layer, tensor_stem))."""
    out: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        out[r["task"]].update(topn_channels(r, topn))
    return dict(out)


def per_language_top_channels(
    rows: list[dict], topn: int
) -> dict[str, set[int]]:
    """Map language -> set of top-N channels (across all rows of that
    language). For this to work, the summary must include `language` on
    each row. The standard analyzer aggregates per (task, layer,
    tensor_stem) so each row has only `task` — language aggregation
    requires the analyzer to emit language-bucketed rows. If the rows
    don't carry `language`, this function returns an empty dict.
    """
    out: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        langs = r.get("languages") or r.get("language_buckets")
        if langs is None:
            continue
        # lang_bucketed analyzer stores {lang: [ch, ...]}
        if isinstance(langs, dict):
            for lang_chan_lists in langs.values():
                for p in lang_chan_lists[:topn]:
                    if isinstance(p, list) and p:
                        out[r.get("language_override", "?")].add(int(p[0]))
    return dict(out)


def pairwise_jaccard_matrix(
    per_unit: dict[str, set[int]],
) -> dict[tuple[str, str], float]:
    """Compute upper-triangular Jaccard overlap for every pair of units."""
    units = sorted(per_unit.keys())
    out: dict[tuple[str, str], float] = {}
    for i, a in enumerate(units):
        for b in units[i + 1 :]:
            out[(a, b)] = jaccard(per_unit[a], per_unit[b])
    return out


def shared_core(
    per_unit: dict[str, set[int]], threshold: int
) -> set[int]:
    """Channels appearing in >= threshold of units."""
    counts: dict[int, int] = defaultdict(int)
    for chset in per_unit.values():
        for c in chset:
            counts[c] += 1
    return {c for c, n in counts.items() if n >= threshold}


def task_specific(
    per_unit: dict[str, set[int]], max_units: int
) -> dict[str, set[int]]:
    """For each unit, channels that appear ONLY in that unit (or appear in
    <= max_units units, restricted to ones where this unit is one of them).
    Default: max_units=1 => channels unique to this unit."""
    # union of all channels across units
    all_chs: set[int] = set()
    for chs in per_unit.values():
        all_chs.update(chs)
    # count per-channel appearance
    counts: dict[int, int] = defaultdict(int)
    for chs in per_unit.values():
        for c in chs:
            counts[c] += 1
    out: dict[str, set[int]] = {}
    for unit, chs in per_unit.items():
        out[unit] = {c for c in chs if counts[c] <= max_units}
    return out


def render_markdown(summary: dict, topn: int) -> str:
    rows = summary["activation_summary"]["rows"]
    by_layer = layer_groups(rows)

    units_tasks = sorted({r["task"] for r in rows})
    n_tasks = len(units_tasks)

    out: list[str] = []
    out.append("# Cross-task activation analysis\n")
    out.append(
        f"Per-(task, layer, tensor_stem) top-{topn} channel comparison "
        f"across {n_tasks} tasks and {len(by_layer)} (layer, tensor_stem) "
        f"groups.\n"
    )
    out.append(
        "Jaccard = |A ∩ B| / |A ∪ B|. Lower overlap = more task-specific "
        "channel selection at that layer.\n"
    )

    # 1) Per-layer pairwise task Jaccard, averaged across layers + std range.
    out.append("\n## Per-layer task-pair Jaccard overlap\n")
    out.append(
        "| layer | tensor_stem | mean Jaccard | min | max | "
        "min pair | max pair |\n"
    )
    out.append("|---|---|---|---|---|---|---|")
    pairwise_per_layer = []
    for (layer, stem), layer_rows in sorted(by_layer.items()):
        per_task = per_task_top_channels(layer_rows, topn)
        if len(per_task) < 2:
            continue
        matrix = pairwise_jaccard_matrix(per_task)
        vals = list(matrix.values())
        if not vals:
            continue
        mean_v = sum(vals) / len(vals)
        min_pair = min(matrix.items(), key=lambda kv: kv[1])
        max_pair = max(matrix.items(), key=lambda kv: kv[1])
        pairwise_per_layer.append((layer, stem, mean_v, min_pair, max_pair))
        out.append(
            f"| {layer} | `{stem}` | {mean_v:.3f} | "
            f"{min_pair[1]:.3f} | {max_pair[1]:.3f} | "
            f"{min_pair[0][0]}↔{min_pair[0][1]} | "
            f"{max_pair[0][0]}↔{max_pair[0][1]} |"
        )

    # 2) Shared core (channels present in >= K of N tasks) per layer.
    out.append("\n## Shared core channels (in ≥half of all tasks)\n")
    out.append("| layer | tensor_stem | # shared | shared core channels |")
    out.append("|---|---|---|---|")
    shared_core_lists = []
    for (layer, stem), layer_rows in sorted(by_layer.items()):
        per_task = per_task_top_channels(layer_rows, topn)
        if len(per_task) < 2:
            continue
        threshold = max(2, len(per_task) // 2 + 1)
        core = sorted(shared_core(per_task, threshold))
        shared_core_lists.append((layer, stem, core))
        out.append(
            f"| {layer} | `{stem}` | {len(core)} | "
            f"{', '.join(f'#{c}' for c in core[:15])}"
            + (f" (+{len(core)-15} more)" if len(core) > 15 else "")
            + " |"
        )

    # 3) Task-specific channels (unique to one task).
    out.append("\n## Task-specific channels (unique to one task)\n")
    out.append("| layer | tensor_stem | task | # uniq | unique channels |")
    out.append("|---|---|---|---|---|")
    task_specific_per_layer = []
    for (layer, stem), layer_rows in sorted(by_layer.items()):
        per_task = per_task_top_channels(layer_rows, topn)
        if len(per_task) < 2:
            continue
        specific = task_specific(per_task, max_units=1)
        for t, chs in sorted(specific.items()):
            if not chs:
                continue
            task_specific_per_layer.append((layer, stem, t, chs))
            out.append(
                f"| {layer} | `{stem}` | {t} | {len(chs)} | "
                f"{', '.join(f'#{c}' for c in sorted(chs)[:15])}"
                + (f" (+{len(chs)-15} more)" if len(chs) > 15 else "")
                + " |"
            )

    # 4) Aggregate conclusions: mean Jaccard across ALL layers, and the
    # task with the most unique channels.
    if pairwise_per_layer:
        flat = [m for _, _, m, _, _ in pairwise_per_layer]
        overall_mean = sum(flat) / len(flat)
        out.append("\n## Aggregate (cross-task)\n")
        out.append(
            f"- Overall mean task-pair Jaccard across all layers: "
            f"**{overall_mean:.3f}**"
        )
        out.append(
            f"- Layers: {min(layer for layer, _, _, _, _ in pairwise_per_layer)}.."
            f"{max(layer for layer, _, _, _, _ in pairwise_per_layer)}"
        )

        # task with most unique channels overall
        uniq_count: dict[str, int] = defaultdict(int)
        for _, _, t, chs in task_specific_per_layer:
            uniq_count[t] += len(chs)
        if uniq_count:
            ranked = sorted(uniq_count.items(), key=lambda kv: -kv[1])
            out.append(
                "\n## Per-task unique channel count (sum across layers)\n"
            )
            out.append("| task | # unique channels |")
            out.append("|---|---|")
            for t, n in ranked:
                out.append(f"| {t} | {n} |")

    # 5) Cross-LANGUAGE comparison (Story 6+): reuses rows_by_language
    # emitted by analyze.py build_summary(). Computes pairwise Jaccard
    # across languages per (layer, tensor_stem).
    lang_rows = summary["activation_summary"].get("rows_by_language") or []
    if lang_rows:
        out.append("\n---\n")
        out.append("# Cross-language activation analysis\n")
        out.append(
            f"Pairwise top-{topn} Jaccard across "
            f"{len({r['language'] for r in lang_rows})} languages "
            f"(redone symmetry to the task analysis above).\n"
        )
        by_layer_lang: dict[tuple[int, str], list[dict]] = defaultdict(list)
        for r in lang_rows:
            by_layer_lang[(int(r["layer"]), r["tensor_stem"])].append(r)
        out.append(
            "\n## Per-layer language-pair Jaccard overlap\n"
        )
        out.append(
            "| layer | tensor_stem | mean Jaccard | min | max | "
            "min pair | max pair |\n"
        )
        out.append("|---|---|---|---|---|---|---|")
        for (layer, stem), layer_rows in sorted(by_layer_lang.items()):
            per_lang: dict[str, set[int]] = defaultdict(set)
            for r in layer_rows:
                per_lang[r["language"]].update(
                    topn_channels(r, topn)
                )
            if len(per_lang) < 2:
                continue
            matrix = pairwise_jaccard_matrix(dict(per_lang))
            vals = list(matrix.values())
            if not vals:
                continue
            mean_v = sum(vals) / len(vals)
            min_pair = min(matrix.items(), key=lambda kv: kv[1])
            max_pair = max(matrix.items(), key=lambda kv: kv[1])
            out.append(
                f"| {layer} | `{stem}` | {mean_v:.3f} | "
                f"{min_pair[1]:.3f} | {max_pair[1]:.3f} | "
                f"{min_pair[0][0]}\u2194{min_pair[0][1]} | "
                f"{max_pair[0][0]}\u2194{max_pair[0][1]} |"
            )

    out.append(
        "\n_Approximation: top-K channel overlap is a "
        "lower-bound interpretability signal, not a full activation trace. "
        "See `GLM52_SESSION_MEMORY.md` for scope._"
    )
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summary",
        required=True,
        help="Path to analyze_moe_trace.py summary JSON (must contain "
        "activation_summary.rows)",
    )
    ap.add_argument(
        "--topn",
        type=int,
        default=10,
        help="Number of top channels per (task, layer) to use for overlap",
    )
    ap.add_argument(
        "--out-md",
        default=None,
        help="Output markdown report path (default: stdout)",
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="Output JSON summary path (default: skip)",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"ERROR: summary file not found: {summary_path}", file=sys.stderr)
        return 2
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("activation_summary", {}).get("rows"):
        print(
            "ERROR: summary has no activation_summary.rows — re-run "
            "analyze_moe_trace.py on a trace set produced with "
            "--trace-activations",
            file=sys.stderr,
        )
        return 3

    md = render_markdown(summary, args.topn)

    if args.out_md:
        Path(args.out_md).write_text(md, encoding="utf-8")
        print(f"wrote {args.out_md}")
    else:
        print(md)

    if args.out_json:
        # Build a JSON version (just per-layer pairwise + shared core)
        rows = summary["activation_summary"]["rows"]
        by_layer = layer_groups(rows)
        json_out = {"topn": args.topn, "per_layer": [], "n_layers": 0}
        for (layer, stem), layer_rows in sorted(by_layer.items()):
            per_task = per_task_top_channels(layer_rows, args.topn)
            if len(per_task) < 2:
                continue
            matrix = pairwise_jaccard_matrix(per_task)
            threshold = max(2, len(per_task) // 2 + 1)
            json_out["per_layer"].append({
                "layer": layer,
                "tensor_stem": stem,
                "n_tasks": len(per_task),
                "pairwise_jaccard": {
                    f"{a}|{b}": v for (a, b), v in matrix.items()
                },
                "mean_jaccard": sum(matrix.values()) / len(matrix) if matrix else 0.0,
                "shared_core": sorted(shared_core(per_task, threshold)),
                "task_specific": {
                    t: sorted(chs)
                    for t, chs in task_specific(per_task, max_units=1).items()
                },
            })
        json_out["n_layers"] = len(json_out["per_layer"])
        Path(args.out_json).write_text(
            json.dumps(json_out, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
