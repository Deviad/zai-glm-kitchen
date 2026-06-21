#!/usr/bin/env python3
"""Aggregate one or more GLM-5.2 MoE trace JSONL files into a Markdown report.

Planned artifact from ``GLM52_TRACE_PLAN.md`` (Phase 1, Story 2):

    common/scripts/analyze_moe_trace.py

Usage::

    python common/scripts/analyze_moe_trace.py \
        --traces "common/traces/glm52-*.jsonl" \
        --out-md  common/reports/glm52_moe_trace_report.md \
        --out-json common/reports/glm52_moe_trace_summary.json \
        --topn 8

The analyzer is backend-agnostic: it consumes the canonical JSONL schema emitted
by the C++ ``trace-moe`` backend OR by ``common/scripts/make_synth_trace.py``.
"""
from __future__ import annotations

import argparse
import sys

from glm52_kitchen.tracing import (
    DEFAULT_K_STEM as RETRIEVAL_DEFAULT_K_STEM,
    DEFAULT_Q_STEM as RETRIEVAL_DEFAULT_Q_STEM,
    DEFAULT_RETRIEVAL_TOPN,
    resolve_traces,
    write_report,
)


def _parse_sentinel_range(spec: str) -> tuple[int, int]:
    """Parse 'START,END' into an inclusive (start, end) tuple.

    Accepts either comma or dash separator; trims whitespace. Used by the
    analyzer to check whether top-N retrieved positions land on the sentinel's
    token-position range (e.g. computed offline from tokenizing the prompt and
    locating the BLUE-FALCON-48217 sentinel).
    """
    s = spec.strip().replace("-", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--sentinel-position-range expects 'START,END' (got {spec!r})"
        )
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"sentinel range bounds must be ints: {e}") from e
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError(
            f"sentinel range must be non-negative and end >= start (got {start}..{end})"
        )
    return start, end


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--traces",
        nargs="+",
        required=True,
        help="One or more trace JSONL files or glob patterns (e.g. 'common/traces/*.jsonl').",
    )
    p.add_argument("--out-md", default="common/reports/glm52_moe_trace_report.md")
    p.add_argument("--out-json", default="common/reports/glm52_moe_trace_summary.json")
    p.add_argument("--topn", type=int, default=8, help="Top-N experts to report per task/language.")
    # Story 5 (re-scoped): MLA retrieval-pattern analysis over activation
    # summaries. Requires traces that were captured with the matching
    # --trace-activations <q>,<k> pair on the C++ side. By default omitted —
    # the section is absent from the report.
    p.add_argument(
        "--retrieval-stems",
        default=None,
        help=(
            "Enable MLA retrieval analysis. Format 'Q_STEM,K_STEM' "
            f"(default if specifier empty: '{RETRIEVAL_DEFAULT_Q_STEM},{RETRIEVAL_DEFAULT_K_STEM}'). "
            "Use together with traces captured via "
            "`--trace-activations Q_STEM,K_STEM` on the C++ tracer."
        ),
    )
    p.add_argument(
        "--retrieval-topn",
        type=int,
        default=DEFAULT_RETRIEVAL_TOPN,
        help=f"Top-N positions to score per (query, layer). Default {DEFAULT_RETRIEVAL_TOPN}.",
    )
    p.add_argument(
        "--sentinel-position-range",
        default=None,
        help=(
            "Optional 'START,END' inclusive token-position range for the sentinel. "
            "The analyzer counts how many (query, layer) retrieval pairs have "
            "≥1 top-N retrieved position inside [START, END]. Computed offline "
            "by tokenizing the prompt and locating the sentinel string."
        ),
    )
    args = p.parse_args(argv)

    sources = resolve_traces(args.traces)
    if not sources:
        print(f"no trace files matched: {args.traces}", file=sys.stderr)
        return 2
    # Story 5 re-scope: parse --retrieval-stems into (q, k) if provided.
    q_stem = None
    k_stem = None
    if args.retrieval_stems is not None:
        spec = args.retrieval_stems.strip()
        if not spec:
            # Empty string: use defaults
            q_stem = RETRIEVAL_DEFAULT_Q_STEM
            k_stem = RETRIEVAL_DEFAULT_K_STEM
        elif "," in spec:
            parts = [p.strip() for p in spec.split(",", 1)]
            q_stem = parts[0] or RETRIEVAL_DEFAULT_Q_STEM
            k_stem = parts[1] or RETRIEVAL_DEFAULT_K_STEM
        else:
            # Single token: assume q default for k
            q_stem = spec
            k_stem = RETRIEVAL_DEFAULT_K_STEM
    sentinel_range = None
    if args.sentinel_position_range is not None:
        sentinel_range = _parse_sentinel_range(args.sentinel_position_range)
    summary = write_report(
        sources,
        args.out_md,
        args.out_json,
        topn=args.topn,
        retrieval_q_stem=q_stem,
        retrieval_k_stem=k_stem,
        retrieval_topn=args.retrieval_topn,
        sentinel_position_range=sentinel_range,
    )
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_json}")
    print(f"records: {summary['n_records']}  sources: {summary['n_sources']}  "
          f"layers: {len(summary['layers'])}  tasks: {len(summary['tasks'])}  "
          f"languages: {len(summary['languages'])}")
    retr = summary.get("retrieval_analysis") or {}
    if retr:
        print(
            f"retrieval_analysis: {retr.get('n_results', 0)} (query, layer) pairs scored "
            f"across {retr.get('n_query_records_seen', 0)} q records / "
            f"{retr.get('n_key_records_seen', 0)} k records "
            f"(q={retr.get('q_stem')}, k={retr.get('k_stem')})"
        )
        if sentinel_range is not None:
            print(f"  sentinel range={sentinel_range}: hits={retr.get('sentinel_hits')} / {retr.get('sentinel_total')} (hit_rate={retr.get('sentinel_hit_rate')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
