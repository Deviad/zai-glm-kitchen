"""Compare multiple trace reports / model runs (Phase 1 Story 8).

Loads summary JSONs produced by :func:`glm52_kitchen.tracing.analyze.write_report`
and renders a side-by-side Markdown comparing expert usage distributions,
flagging missing or renamed experts, and surfacing speed metrics.  Carries the
hardcoded caveat that the REAP37 MLX compat artifact is **invalid** for quality
comparison until proper IndexShare support lands (see ``REAP37_EXPERIMENTS.md``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_REAP_INVALID_HINT = (
    "REAP37 MLX indexer-compat traces are marked INVALID for quality comparison: "
    "stock mlx-lm lacks IndexShare support; see REAP37_EXPERIMENTS.md."
)


def load_summary(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _pooled_top(summary: dict[str, Any]) -> dict[tuple[int, int], int]:
    """Flatten top_experts_by_task_layer into {(expert, layer): count}."""
    out: dict[tuple[int, int], int] = {}
    for _task, per_layer in summary.get("top_experts_by_task_layer", {}).items():
        for layer_str, rows in per_layer.items():
            layer = int(layer_str)
            for e, n in rows:
                out[(int(e), layer)] = out.get((int(e), layer), 0) + int(n)
    return out


def compare(
    summaries: list[tuple[str, dict[str, Any]]],
    topn: int = 8,
) -> dict[str, Any]:
    labels = [label for label, _ in summaries]
    union_experts: set[int] = set()
    pooled = {}
    per_label_top_global: dict[str, list[tuple[int, int]]] = {}
    for label, s in summaries:
        flat = _pooled_top(s)
        # per-layer top pooled across layers → global top per label
        per_layer: dict[int, int] = {}
        for (e, _layer), n in flat.items():
            per_layer[e] = per_layer.get(e, 0) + n
        pooled[label] = per_layer
        union_experts |= set(per_layer.keys())
        per_label_top_global[label] = sorted(per_layer.items(), key=lambda kv: -kv[1])[:topn]

    # missing experts: in some labels but absent in others
    missing: dict[str, list[int]] = {}
    for label in labels:
        present = set(pooled[label].keys())
        missing[label] = sorted(union_experts - present)

    expert_count_changes: dict[str, int] = {}
    n_expert_total_values = {
        s.get("n_expert_total") for _, s in summaries if s.get("n_expert_total")
    }
    if len(n_expert_total_values) > 1:
        for label, s in summaries:
            expert_count_changes[label] = s.get("n_expert_total")

    # speed
    speed: dict[str, dict[str, float | None]] = {}
    for label, s in summaries:
        runs = s.get("sources", [])
        speeds = [r.get("perf_gen_per_sec") for r in runs if r.get("perf_gen_per_sec")]
        speed[label] = {
            "n_runs": len(runs),
            "mean_gen_per_sec": round(sum(speeds) / len(speeds), 2) if speeds else None,
        }

    return {
        "schema_version": 1,
        "labels": labels,
        "topn": topn,
        "global_top_experts_by_label": {
            lbl: [[int(e), int(n)] for e, n in rows] for lbl, rows in per_label_top_global.items()
        },
        "missing_experts_by_label": {lbl: missing[lbl] for lbl in labels},
        "expert_total_changes": expert_count_changes,
        "speed_by_label": speed,
        "caveats": [_REAP_INVALID_HINT if any("reap" in lbl.lower() for lbl in labels) else None],
    }


def render_markdown(comp: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GLM-5.2 Trace Baseline Comparison")
    lines.append("")
    labels = comp["labels"]
    lines.append(f"Comparing **{len(labels)}** run(s): `{', '.join(labels)}`")
    lines.append("")
    lines.append("## Global top experts by run")
    headers = ["expert"] + labels
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    # union of top experts across labels, sorted by max prevalence
    all_experts: set[int] = set()
    for rows in comp["global_top_experts_by_label"].values():
        all_experts |= {e for e, _ in rows}
    prevalence: dict[int, int] = {}
    counts_by_label = {
        lbl: {e: n for e, n in comp["global_top_experts_by_label"][lbl]} for lbl in labels
    }
    for e in all_experts:
        prevalence[e] = sum(counts_by_label[lbl].get(e, 0) for lbl in labels)
    for e in sorted(all_experts, key=lambda x: -prevalence[x])[: comp["topn"]]:
        row = [str(e)] + [str(counts_by_label[lbl].get(e, 0)) for lbl in labels]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Missing experts (present in some runs, absent in others)")
    lines.append("| run | n_missing | sample missing experts |")
    lines.append("|---|---|---|")
    for lbl in labels:
        miss = comp["missing_experts_by_label"].get(lbl, [])
        sample = ", ".join(f"#{e}" for e in miss[:12])
        lines.append(f"| {lbl} | {len(miss)} | {sample} |")
    lines.append("")

    if comp.get("expert_total_changes"):
        lines.append("## Expert-count changes")
        lines.append("| run | n_expert_total |")
        lines.append("|---|---|")
        for lbl, v in comp["expert_total_changes"].items():
            lines.append(f"| {lbl} | {v} |")
        lines.append("")

    lines.append("## Speed")
    lines.append("| run | n_runs | mean gen/s |")
    lines.append("|---|---|---|")
    for lbl in labels:
        s = comp["speed_by_label"].get(lbl, {})
        lines.append(f"| {lbl} | {s.get('n_runs', 0)} | {s.get('mean_gen_per_sec')} |")
    lines.append("")

    for c in comp.get("caveats", []):
        if c:
            lines.append(f"> ⚠️ {c}")
            lines.append("")

    return "\n".join(lines) + "\n"


def write_comparison(
    summary_paths: list[tuple[str, str]],
    out_md: str | os.PathLike[str],
    out_json: str | os.PathLike[str],
    *,
    topn: int = 8,
) -> dict[str, Any]:
    summaries = [(label, load_summary(p)) for label, p in summary_paths]
    comp = compare(summaries, topn=topn)
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    md = render_markdown(comp)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(comp, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return comp


__all__ = ["compare", "render_markdown", "load_summary", "write_comparison"]
