#!/usr/bin/env python3
"""Generate synthetic GLM-5.2 MoE traces from the smoke suite.

Used to test the analyzer + writer and to produce a sample report without
loading the 232GB model.  The synthetic records follow the EXACT same JSONL
schema as the real C++ ``trace-moe`` backend, so analyzer results are faithful
to what a live run would produce structurally (only the distribution is fake).

Usage::

    python common/scripts/make_synth_trace.py \
        --suite prompts/tracing/glm52_trace_smoke_suite.expanded.jsonl \
        --out-dir traces/synth \
        --limit 20

    # then analyze:
    python common/scripts/analyze_moe_trace.py \
        --traces "traces/synth/*.jsonl" \
        --out-md reports/glm52_moe_trace_report.md \
        --out-json reports/glm52_moe_trace_summary.json
"""
from __future__ import annotations

import argparse
import os
import sys

from glm52_kitchen.tracing import iter_suite, write_synth_trace


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--suite",
        default="prompts/tracing/glm52_trace_smoke_suite.expanded.jsonl",
        help="Expanded smoke-suite JSONL (one prompt record per line).",
    )
    p.add_argument("--out-dir", default="traces/synth", help="Directory to write trace JSONL files.")
    p.add_argument("--limit", type=int, default=0, help="Max records to emit (0 = all).")
    p.add_argument("--n-prefill", type=int, default=24)
    p.add_argument("--n-gen", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=79)
    p.add_argument("--n-expert", type=int, default=256)
    p.add_argument("--n-expert-used", type=int, default=4)
    args = p.parse_args(argv)

    if not os.path.exists(args.suite):
        print(f"suite not found: {args.suite}", file=sys.stderr)
        return 2
    os.makedirs(args.out_dir, exist_ok=True)

    n = 0
    for rec in iter_suite(args.suite):
        if args.limit and n >= args.limit:
            break
        test_id = rec.get("test_id", f"rec{n}")
        language = rec.get("language", "en")
        out_path = os.path.join(args.out_dir, f"synth-{test_id}-{language}.jsonl")
        if os.path.exists(out_path):
            # Skip so reruns are idempotent; match the writer's no-overwrite contract.
            n += 1
            continue
        write_synth_trace(
            rec,
            out_path,
            n_prefill=args.n_prefill,
            n_gen=args.n_gen,
            n_layers=args.n_layers,
            n_expert=args.n_expert,
            n_expert_used=args.n_expert_used,
        )
        n += 1
    print(f"wrote {n} synthetic trace files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
