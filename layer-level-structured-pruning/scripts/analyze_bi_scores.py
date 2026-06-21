#!/usr/bin/env python3
"""Aggregate per-layer Block Influence (BI) scores from trace-moe JSONL.

Inputs: one or more .jsonl files containing ``block_influence`` records
(emitted by llama-trace-moe when ``--trace-activations l_out`` is set, on
every layer regardless of ``--trace-activation-stride``).

Computes:
  * per-(layer, prompt) mean BI across tokens (sum of bi_score / n_tokens)
  * per-layer mean BI across prompts    (mean of prompt means)
  * per-layer median BI across prompts
  * per-layer std BI across prompts
  * rank: ascending BI  (lowest = most redundant = best ShortGPT candidate)

Outputs:
  * A markdown report at --report  (default reports/glm52_shortgpt_bi_scores.md)
  * A JSON plan at   --plan    (default reports/glm52_shortgpt_bi_scores.json)
      with ``top_n`` lowest-BI layers identified for pruning.

Usage::

    python3 scripts/analyze_bi_scores.py \\
        --traces 'traces/shortgpt_bi/calib/*.jsonl' \\
        --top-n 16 \\
        --report reports/glm52_shortgpt_bi_scores.md \\
        --plan   reports/glm52_shortgpt_bi_scores.json
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_bi_records(patterns: list[str]) -> list[dict]:
    """Read all block_influence records from files matching the patterns."""
    recs: list[dict] = []
    files: set[str] = set()
    for pat in patterns:
        files.update(glob.glob(pat))
    if not files:
        raise SystemExit(f"no files matched {patterns}")
    for fp in sorted(files):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") != "block_influence":
                    continue
                recs.append(r)
    return recs


def aggregate(recs: list[dict]) -> dict:
    """Aggregate BI records. Returns a dict keyed by layer with stats."""
    # per-layer per-(run_id) lists of bi_score
    per_layer_per_prompt: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_layer_per_prompt_cos: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in recs:
        L = r["layer"]
        rid = r.get("run_id", "?")
        # Use (run_id, test_id, language) as the per-prompt key so the same
        # trace file containing multiple prompts (rare) gets distinct entries.
        key = f"{rid}|{r.get('test_id', '')}|{r.get('language', '')}"
        per_layer_per_prompt[L][key].append(float(r["bi_score"]))
        per_layer_per_prompt_cos[L][key].append(float(r["cos_sim"]))

    # Per-prompt mean BI per layer
    per_layer_prompt_mean_bi: dict[int, dict[str, float]] = {}
    per_layer_prompt_mean_cos: dict[int, dict[str, float]] = {}
    for L, prompts in per_layer_per_prompt.items():
        per_layer_prompt_mean_bi[L] = {}
        per_layer_prompt_mean_cos[L] = {}
        for key, scores in prompts.items():
            if not scores:
                continue
            per_layer_prompt_mean_bi[L][key] = statistics.fmean(scores)
            per_layer_prompt_mean_cos[L][key] = statistics.fmean(
                per_layer_per_prompt_cos[L][key]
            )

    # Per-layer aggregated stats
    layer_stats: list[dict] = []
    for L in sorted(per_layer_prompt_mean_bi.keys()):
        prompt_means = list(per_layer_prompt_mean_bi[L].values())
        cos_means = list(per_layer_prompt_mean_cos[L].values())
        if not prompt_means:
            continue
        n_prompts = len(prompt_means)
        mean_bi = statistics.fmean(prompt_means)
        median_bi = statistics.median(prompt_means)
        stdev_bi = statistics.stdev(prompt_means) if n_prompts >= 2 else 0.0
        mean_cos = statistics.fmean(cos_means)
        layer_stats.append(
            {
                "layer": L,
                "n_prompts": n_prompts,
                "mean_bi": mean_bi,
                "median_bi": median_bi,
                "stdev_bi": stdev_bi,
                "min_prompt_bi": min(prompt_means),
                "max_prompt_bi": max(prompt_means),
                "mean_cos_sim": mean_cos,
            }
        )

    # Rank ascending by mean BI (most redundant = lowest BI = rank 1)
    layer_stats.sort(key=lambda d: d["mean_bi"])
    for rank, s in enumerate(layer_stats, start=1):
        s["redundancy_rank"] = rank

    return {
        "layer_stats_by_redundancy": layer_stats,
        "n_records": len(recs),
        "n_layers": len(layer_stats),
    }


def write_report(agg: dict, plan: dict, report_path: Path) -> None:
    lines: list[str] = []
    lines.append("# GLM-5.2 ShortGPT BI scoring report\n")
    lines.append(
        "Block Influence (BI) = 1 - cos(h_in, h_out) where h_in = previous "
        "layer's\nresidual and h_out = this layer's residual (per token, "
        "averaged over prompts).\nLower BI ⟹ more redundant layer ⟹ better "
        "prune candidate (ShortGPT heuristic).\n"
    )
    lines.append(f"- Total BI records ingested: **{agg['n_records']}**")
    lines.append(f"- Distinct layers scored: **{agg['n_layers']}**")
    lines.append(
        f"- Prune target: **{plan['n_to_prune']}** lowest-BI layers "
        f"({plan['prune_fraction']*100:.0f}% of {plan['n_layers_baseline']} "
        f"normal layers)\n"
    )
    lines.append("## Prune plan\n")
    lines.append(
        f"Layers to drop (original indices, by ascending BI): "
        f"**{plan['layers_to_drop']}**\n"
    )
    lines.append(
        f"Resulting model: {plan['n_layers_baseline']}→"
        f"{plan['n_layers_baseline'] - plan['n_to_prune']} normal layers\n"
    )
    lines.append("## Per-layer BI (ascending = most redundant first)\n")
    lines.append(
        "| rank | layer | mean_BI | median_BI | stdev_BI | mean_cos_sim | "
        "n_prompts |"
    )
    lines.append("|------|-------|---------|-----------|----------|--------------|-----------|")
    for s in agg["layer_stats_by_redundancy"]:
        lines.append(
            f"| {s['redundancy_rank']} | blk.{s['layer']} "
            f"| {s['mean_bi']:.6f} | {s['median_bi']:.6f} "
            f"| {s['stdev_bi']:.6f} | {s['mean_cos_sim']:.6f} "
            f"| {s['n_prompts']} |"
        )
    report_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {report_path}")


def write_plan(agg: dict, top_n: int, plan_path: Path, *, max_contiguous: int = 1) -> dict:
    """Build the prune plan from the top-N lowest-BI ranking.

    Default `max_contiguous=1`: prohibits any 2 consecutive layer drops
    (i.e. drops must be spaced by ≥1 kept layer between them). This is the
    empirically-validated safe zone for GLM-5.2 — Phase 8's spaced-N=12 plan
    (no adjacent drops) PASSED both baselines, while the contiguous-N=16 plan
    (14 consecutive drops in L51-L64) COLLAPSED into merge-sort repetition.
    See GLM52_SESSION_MEMORY.md 'Phase 8 follow-up' for the forensic record.

    If you genuinely want to drop an adjacent pair (e.g. for an ablation study),
    pass max_contiguous=2; pass 0 to disable the cap entirely.
    """
    baseline_layers = agg["n_layers"]  # 77 (1..77)
    # Pass 1: greedily take the lowest-BI layer, skipping any candidate
    # that would extend an existing contiguous run beyond max_contiguous.
    # This generates a SPACED drop plan (adjacent drops forbidden unless
    # the cap allows). Layers with no BI score (layer 0) are always kept.
    n_target = min(top_n, baseline_layers // 2)
    picked: list[int] = []
    capped_warnings = 0
    for stat in agg["layer_stats_by_redundancy"]:
        if len(picked) >= n_target:
            break
        L = stat["layer"]
        if L == 0:
            continue  # no BI score (no previous layer to compare against)
        if max_contiguous > 0:
            # Count contiguous adjacency to the already-picked set. If adding
            # L would create a run longer than max_contiguous, skip.
            adjacent_count = sum(1 for p in picked if abs(L - p) <= 1)
            # If L is adjacent to one or more existing picks ('extends a run'),
            # check the current run length that L would extend.
            if adjacent_count > 0:
                # Walk left and right through already-picked layers
                # starting from L's neighborhood.
                run_length = 1  # L itself
                # Extend left (L-1, L-2, ...) while picked
                cur = L - 1
                while cur in picked:
                    run_length += 1
                    cur -= 1
                # Extend right (L+1, L+2, ...) while picked
                cur = L + 1
                while cur in picked:
                    run_length += 1
                    cur += 1
                if run_length > max_contiguous:
                    capped_warnings += 1
                    continue
        picked.append(L)
    picked.sort()
    layers_to_drop = picked
    n_to_prune = len(layers_to_drop)

    # Build the renumbering map: for each kept original idx, its new index in
    # the pruned contiguous sequence.
    kept_original = [L for L in range(0, baseline_layers + 1) if L not in layers_to_drop]
    # kept_original[0] is layer 0 (the embedding-output layer, no BI scored);
    # it always stays. The renumbering maps each kept original idx → its
    # position in the kept list.
    renumber_map = {orig: new for new, orig in enumerate(kept_original)}
    plan = {
        "n_layers_baseline": baseline_layers,
        "prune_fraction": n_to_prune / baseline_layers if baseline_layers else 0.0,
        "n_to_prune": n_to_prune,
        "layers_to_drop": layers_to_drop,
        "layers_kept": kept_original,
        "renumber_map": renumber_map,  # original_idx -> new_idx
        "block_count_new": len(kept_original) + 1,  # +1 because MTP layer still present at blk.78→blk.N+1
        "max_contiguous_drops": max_contiguous,
        "n_skipped_due_to_contiguity_cap": capped_warnings,
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"  wrote {plan_path}")
    if capped_warnings:
        print(f"  ({capped_warnings} candidates skipped by max_contiguous={max_contiguous} cap;")
        print("   pass --max-contiguous-drops 0 to disable)")
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--traces",
        nargs="+",
        required=True,
        help="One or more glob patterns matching trace .jsonl files.",
    )
    ap.add_argument("--top-n", type=int, default=16)
    ap.add_argument(
        "--max-contiguous-drops",
        type=int,
        default=1,
        help="Maximum length of any run of consecutive layer indices in "
             "the drop set. Default 1 (no two consecutive drops — the empirically-"
             "validated safe zone for GLM-5.2; Phase 8 spaced-N=12 with no "
             "adjacency PASSED baselines, contiguous-N=16 COLLAPSED). Pass 0 to "
             "disable the cap entirely.",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("reports/glm52_shortgpt_bi_scores.md"),
    )
    ap.add_argument(
        "--plan",
        type=Path,
        default=Path("reports/glm52_shortgpt_bi_scores.json"),
    )
    args = ap.parse_args()

    recs = load_bi_records(args.traces)
    print(f"  loaded {len(recs)} block_influence records")
    agg = aggregate(recs)
    plan = write_plan(agg, args.top_n, args.plan, max_contiguous=args.max_contiguous_drops)
    write_report(agg, plan, args.report)

    # Headline stdout
    print()
    print(
        f"  Top-{plan['n_to_prune']} lowest-BI prune candidates "
        f"({plan['prune_fraction']*100:.0f}% of {plan['n_layers_baseline']} normal layers):"
    )
    print(f"    {plan['layers_to_drop']}")
    print(f"  new block_count: {plan['block_count_new']}")


if __name__ == "__main__":
    main()
