#!/usr/bin/env python3
"""Expand the multilingual GLM-5.2 trace smoke suite to one JSONL record per prompt.

Examples:
  python common/scripts/expand_smoke_suite.py \
    --suite prompts/tracing/glm52_trace_smoke_suite.json \
    --out prompts/tracing/glm52_trace_smoke_suite.expanded.jsonl

  python common/scripts/expand_smoke_suite.py --language en --domain coding
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        default="prompts/tracing/glm52_trace_smoke_suite.json",
        help="Path to structured smoke suite JSON",
    )
    parser.add_argument(
        "--out",
        default="prompts/tracing/glm52_trace_smoke_suite.expanded.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--language", action="append", help="Language code filter; repeatable")
    parser.add_argument("--domain", action="append", help="Domain filter; repeatable")
    args = parser.parse_args()

    suite_path = Path(args.suite)
    out_path = Path(args.out)
    data: dict[str, Any] = json.loads(suite_path.read_text(encoding="utf-8"))

    language_filter = set(args.language or [])
    domain_filter = set(args.domain or [])

    langs = data["meta"]["languages"]
    records: list[dict[str, Any]] = []

    for test in data["tests"]:
        if domain_filter and test["domain"] not in domain_filter:
            continue
        for lang, prompt in test["prompts"].items():
            if language_filter and lang not in language_filter:
                continue
            script = "Han" if lang == "zh" else "Latin"
            record = {
                "schema_version": data["meta"]["schema_version"],
                "suite_name": data["meta"]["name"],
                "test_id": test["id"],
                "domain": test["domain"],
                "task_type": test["task_type"],
                "difficulty": test["difficulty"],
                "answer_style": test["answer_style"],
                "language": lang,
                "language_name": langs[lang],
                "script": script,
                "prompt_family": test["domain"],
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
            }
            records.append(record)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"wrote {len(records)} records to {out_path}")
    by_domain: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    for r in records:
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1
        by_lang[r["language"]] = by_lang.get(r["language"], 0) + 1
    print("by_domain", json.dumps(by_domain, sort_keys=True))
    print("by_language", json.dumps(by_lang, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
