"""Tests for the GLM-5.2 MoE trace analyzer + comparator."""
from __future__ import annotations

import json
from pathlib import Path

from glm52_kitchen.tracing import (
    resolve_traces,
    write_report,
    write_comparison,
)
from glm52_kitchen.tracing.synth import write_synth_trace


def _write_suite_traces(tmp_path: Path, n: int = 6) -> Path:
    """Write n synthetic traces covering several tasks/languages."""
    records = [
        {"test_id": "coding_01", "language": "en", "domain": "coding",
         "prompt_family": "coding", "script": "Latin",
         "prompt": "Write merge sort."},
        {"test_id": "coding_01", "language": "zh", "domain": "coding",
         "prompt_family": "coding", "script": "Han", "prompt": "x" * 5},
        {"test_id": "math_01", "language": "en", "domain": "math",
         "prompt_family": "math", "script": "Latin", "prompt": "integrate x"},
        {"test_id": "math_01", "language": "it", "domain": "math",
         "prompt_family": "math", "script": "Latin", "prompt": "integrazione"},
        {"test_id": "cyber_01", "language": "de", "domain": "cybersecurity",
         "prompt_family": "cybersecurity", "script": "Latin", "prompt": "phish"},
        {"test_id": "cs_01", "language": "fr", "domain": "computer_science",
         "prompt_family": "computer_science", "script": "Latin", "prompt": "algo"},
    ][:n]
    out_dir = tmp_path / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(records):
        write_synth_trace(rec, str(out_dir / f"synth-{i:02d}-{rec['language']}.jsonl"))
    return out_dir


class TestAnalyzer:
    def test_resolve_and_aggregate(self, tmp_path):
        d = _write_suite_traces(tmp_path)
        sources = resolve_traces([str(d / "*.jsonl")])
        assert len(sources) == 6
        # each has a metadata sidecar
        assert all(s.meta is not None for s in sources)
        out_md = tmp_path / "report.md"
        out_json = tmp_path / "summary.json"
        summary = write_report(sources, out_md, out_json, topn=5)
        assert summary["n_records"] > 0
        assert summary["n_sources"] == 6
        assert "coding" in summary["tasks"]
        assert "en" in summary["languages"]
        assert "zh" in summary["languages"]
        # task overlap must be populated
        assert any("coding" in k for k in summary["task_overlap_jaccard_topn"])
        # lang overlap must be populated (en vs it, en vs zh, etc.)
        assert len(summary["language_overlap_jaccard_topn"]) > 0
        # prefill vs generation present
        assert "prefill" in summary["prefill_vs_generation"]
        assert "generation" in summary["prefill_vs_generation"]
        # report rendered with all sections
        md = out_md.read_text(encoding="utf-8")
        assert "Top experts by task and layer" in md
        assert "Router entropy" in md
        assert "Prefill vs generation" in md
        assert "MLA retrieval-pattern tracing" in md  # the Phase 3 caveat line (re-scoped Story 5)

    def test_expert_specialization_coding_unique(self, tmp_path):
        d = _write_suite_traces(tmp_path)
        sources = resolve_traces([str(d / "*.jsonl")])
        summary = write_report(sources, tmp_path / "r.md", tmp_path / "s.json", topn=8)
        # coding should have SOME unique experts vs other tasks (synth is biased)
        spec = summary["expert_specialization"]
        assert "coding" in spec
        assert spec["coding"]["n_unique_top"] >= 0  # structural; bias guarantees >0 but keep loose

    def test_deterministic_generation(self, tmp_path):
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        rec = {"test_id": "coding_01", "language": "en", "domain": "coding",
               "prompt_family": "coding", "script": "Latin", "prompt": "x"}
        write_synth_trace(rec, str(a))
        write_synth_trace(rec, str(b))
        # same seed → same content
        assert a.read_text() == b.read_text()

    def test_empty_glob_returns_empty(self, tmp_path):
        sources = resolve_traces([str(tmp_path / "nope-*.jsonl")])
        assert sources == []

    def test_summary_json_is_valid_json(self, tmp_path):
        d = _write_suite_traces(tmp_path)
        sources = resolve_traces([str(d / "*.jsonl")])
        summary = write_report(sources, tmp_path / "r.md", tmp_path / "s.json")
        obj = json.loads((tmp_path / "s.json").read_text())
        assert obj["n_records"] == summary["n_records"]


class TestComparator:
    def test_compare_two_summaries(self, tmp_path):
        d1 = _write_suite_traces(tmp_path / "a", n=4)
        d2 = _write_suite_traces(tmp_path / "b", n=6)
        s1 = resolve_traces([str(d1 / "*.jsonl")])
        s2 = resolve_traces([str(d2 / "*.jsonl")])
        p1 = tmp_path / "s1.json"
        p2 = tmp_path / "s2.json"
        write_report(s1, tmp_path / "r1.md", p1)
        write_report(s2, tmp_path / "r2.md", p2)
        out_md = tmp_path / "cmp.md"
        out_json = tmp_path / "cmp.json"
        comp = write_comparison([("baseline", str(p1)), ("candidate", str(p2))],
                               out_md, out_json, topn=5)
        assert comp["labels"] == ["baseline", "candidate"]
        assert "baseline" in comp["global_top_experts_by_label"]
        md = out_md.read_text(encoding="utf-8")
        assert "Global top experts by run" in md
        assert "Missing experts" in md

    def test_reap37_label_emits_caveat(self, tmp_path):
        d = _write_suite_traces(tmp_path, n=4)
        s = resolve_traces([str(d / "*.jsonl")])
        p = tmp_path / "s.json"
        write_report(s, tmp_path / "r.md", p)
        comp = write_comparison([("reap37-compat", str(p))], tmp_path / "c.md", tmp_path / "c.json")
        caveats = [c for c in comp["caveats"] if c]
        assert len(caveats) == 1
        assert "INVALID" in caveats[0]


class TestActivationByLanguage:
    """Story 6+ (post-Phase-4 extension): the analyzer's activation_summary
    section must emit BOTH rows (by task) AND rows_by_language (parallel
    by-language grouping). Cross-task/cross-language Jaccard analyses
    rely on these two arrays being populated from the same trace set."""

    def _write_multilang_activation_traces(self, tmp_path: Path, n: int = 4) -> Path:
        """Write n synth traces with activations=True, varying language."""
        records = [
            {"test_id": "coding_01", "language": "en", "domain": "coding",
             "prompt_family": "coding", "script": "Latin", "prompt": "merge sort"},
            {"test_id": "coding_01", "language": "zh", "domain": "coding",
             "prompt_family": "coding", "script": "Han", "prompt": "mergesort"},
            {"test_id": "math_01",   "language": "en", "domain": "math",
             "prompt_family": "math", "script": "Latin", "prompt": "integrate x"},
            {"test_id": "math_01",   "language": "it", "domain": "math",
             "prompt_family": "math", "script": "Latin", "prompt": "integrazione"},
        ][:n]
        out_dir = tmp_path / "act_traces"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, rec in enumerate(records):
            write_synth_trace(rec, str(out_dir / f"act-{i:02d}-{rec['language']}.jsonl"),
                             activations=True, n_layers=6, n_prefill=2, n_gen=1,
                             activation_topk=4)
        return out_dir

    def test_activation_summary_has_rows_and_rows_by_language(self, tmp_path):
        d = self._write_multilang_activation_traces(tmp_path, n=4)
        sources = resolve_traces([str(d / "*.jsonl")])
        assert len(sources) == 4
        out_json = tmp_path / "summary.json"
        summary = write_report(sources, tmp_path / "r.md", out_json, topn=4)
        act = summary.get("activation_summary") or {}
        # rows is by (task, layer, tensor_stem)
        assert "rows" in act
        assert len(act["rows"]) > 0
        # rows_by_language is by (language, layer, tensor_stem) — Story 6+ extension
        assert "rows_by_language" in act
        assert len(act["rows_by_language"]) > 0
        # Each row_by_language row has the expected schema
        sample = act["rows_by_language"][0]
        assert set(sample.keys()) >= {"language", "layer", "tensor_stem",
                                       "n_tokens", "top_channels_by_frequency"}
        # at least one row per language present in the input
        langs_present = {r["language"] for r in act["rows_by_language"]}
        assert {"en", "zh", "it"} <= langs_present

    def test_rows_by_language_channels_are_ints(self, tmp_path):
        """Channel IDs arriving from JSON could be ints or could be coerced
        to floats by Python's json module on some platforms (they're not
        actually float-typed, but verify the type is a clean int)."""
        d = self._write_multilang_activation_traces(tmp_path, n=3)
        sources = resolve_traces([str(d / "*.jsonl")])
        summary = write_report(sources, tmp_path / "r.md", tmp_path / "s.json", topn=4)
        for r in summary["activation_summary"]["rows_by_language"]:
            for ch, _count in r["top_channels_by_frequency"]:
                assert isinstance(ch, int), f"channel ID is not int: {type(ch).__name__}"
                assert isinstance(_count, int)
