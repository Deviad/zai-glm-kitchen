#!/usr/bin/env python3
"""Compare two or more GLM-5.2 trace summaries side-by-side (Story 8).

Planned artifact from ``GLM52_TRACE_PLAN.md`` (Phase 1, Story 8):

    common/scripts/compare_trace_reports.py

Usage::

    python common/scripts/compare_trace_reports.py \
        --label baseline  --summary reports/glm52_moe_trace_summary.json \
        --label reap37    --summary reports/reap37_moe_trace_summary.json \
        --out-md  reports/glm52_baseline_vs_reap37.md \
        --out-json reports/glm52_baseline_vs_reap37.json

Carries the hardcoded caveat that REAP37 MLX indexer-compat traces are INVALID
for quality comparison until proper IndexShare support lands.
"""
from __future__ import annotations

import argparse
import sys

from glm52_kitchen.tracing import write_comparison


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--label",
        action="append",
        required=True,
        help="Run/model label. Repeat once per --summary. First --label pairs with first --summary, etc.",
    )
    p.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Path to a summary JSON from analyze_moe_trace.py. Repeat once per --label.",
    )
    p.add_argument("--out-md", default="reports/glm52_trace_comparison.md")
    p.add_argument("--out-json", default="reports/glm52_trace_comparison.json")
    p.add_argument("--topn", type=int, default=8)
    args = p.parse_args(argv)

    if len(args.label) != len(args.summary):
        print("--label and --summary must be repeated the same number of times", file=sys.stderr)
        return 2

    pairs = list(zip(args.label, args.summary))
    comp = write_comparison(pairs, args.out_md, args.out_json, topn=args.topn)
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_json}")
    print(f"labels: {comp['labels']}  caveats: {[c for c in comp['caveats'] if c]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
