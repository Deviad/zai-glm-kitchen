"""Tests for the channel-focus analyzer (Phase 6 tool)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "common" / "scripts" / "analyze_channel_focus.py"


def _write_focus_trace(path: Path, *, target_channel: int = 42, run_id: str = "t1"):
    """Write a tiny synthetic activation trace for testing the focus analyzer.

    Three records:
      - record 0: target is rank-1 (highest mag in top-K), prefill, layer 0
      - record 1: target is in top-K but not rank-1, prefill, layer 6
      - record 2: target is NOT in top-K (analyzer should skip it)
    """
    rec0 = {
        "event": "activation_summary", "schema_version": 1, "run_id": run_id,
        "model": "test", "phase": "prefill", "token_index": 0, "layer": 0,
        "task_label": "coding", "language": "en", "script": "Latin",
        "prompt_family": "coding", "test_id": "test_act",
        "tensor_stem": "l_out", "n_channels": 100, "topk": 3,
        "top_k_channels": [[target_channel, 5.0], [1, 2.0], [2, 1.0]],
        "l2_norm": 6.0, "mean": 0.0, "std": 1.0, "max_abs": 5.0,
    }
    rec1 = {
        "event": "activation_summary", "schema_version": 1, "run_id": run_id,
        "model": "test", "phase": "prefill", "token_index": 5, "layer": 6,
        "task_label": "math", "language": "en", "script": "Latin",
        "prompt_family": "math", "test_id": "test_act",
        "tensor_stem": "l_out", "n_channels": 100, "topk": 3,
        "top_k_channels": [[1, 9.0], [target_channel, 4.0], [2, 1.0]],
        "l2_norm": 10.0, "mean": 0.0, "std": 2.0, "max_abs": 9.0,
    }
    rec2 = {
        "event": "activation_summary", "schema_version": 1, "run_id": run_id,
        "model": "test", "phase": "generation", "token_index": 10, "layer": 6,
        "task_label": "physics", "language": "zh", "script": "Han",
        "prompt_family": "physics", "test_id": "test_act",
        "tensor_stem": "l_out", "n_channels": 100, "topk": 3,
        "top_k_channels": [[5, 1.0], [6, 2.0], [7, 3.0]],
        "l2_norm": 4.0, "mean": 0.0, "std": 0.5, "max_abs": 3.0,
    }
    with open(path, "w", encoding="utf-8") as fh:
        for r in (rec0, rec1, rec2):
            fh.write(json.dumps(r) + "\n")
    # Meta sidecar with prompt_token_count (=10), so position-bucketing can work
    meta = {
        "schema_version": 1, "run_id": run_id, "model": "test",
        "task_label": "coding", "language": "en",
        "prompt_token_count": 10, "gen_token_count": 1,
    }
    with open(str(path) + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)


class TestChannelFocus:
    def test_find_channel_in_topk_importable(self):
        sys.path.insert(0, str(SCRIPT.parent))
        try:
            from analyze_channel_focus import find_channel_in_topk
        finally:
            sys.path.pop(0)
        # rank-1 case
        top = [[42, 5.0], [1, 2.0], [2, 1.0]]
        assert find_channel_in_topk(top, 42) == (1, 5.0)
        # rank-2 case
        assert find_channel_in_topk(top, 1) == (2, 2.0)
        # not present
        assert find_channel_in_topk(top, 99) is None

    def test_cli_end_to_end(self, tmp_path):
        _write_focus_trace(tmp_path / "test.jsonl", target_channel=42)
        out_md = tmp_path / "out.md"
        out_json = tmp_path / "out.json"
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--traces", str(tmp_path / "*.jsonl"),
             "--channel", "42",
             "--out-md", str(out_md),
             "--out-json", str(out_json)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, f"stdout={result.stdout}, stderr={result.stderr}"
        s = json.loads(out_json.read_text(encoding="utf-8"))
        # Three activation records scanned; channel #42 appears in 2 of them
        assert s["n_activation_records_scanned"] == 3
        assert s["n_records_with_channel"] == 2
        # One of those is rank-1
        assert s["n_records_rank1"] == 1
        assert s["rank_distribution_overall"] == {"1": 1, "2": 1}
        # Co-fire: rank-1 record is rec0 with target=42; the OTHER channels
        # present (1, 2) should appear in cofire_when_rank1_top20.
        co = s["cofire_channels_when_rank1_top20"]
        assert co.get("1") == 1
        assert co.get("2") == 1
        # Position buckets: rank-1 was rec 0 (token_index=0, prompt_len=10),
        # frac=0.0, falls in "0-10%" bucket for coding prefill.
        pos = s["position_buckets_by_phase_task_rank1"]
        assert pos.get("prefill|coding", {}).get("0-10%") == 1
        # Layer table covers layers 0 and 6
        layers = {row["layer"] for row in s["per_layer_magnitude"]}
        assert layers == {0, 6}
        # Task x phase table covers (coding, prefill), (math, prefill)
        # because rec2 (physics) had no target in top-K
        tp = {(row["task"], row["phase"]) for row in s["task_phase_magnitude"]}
        assert ("coding", "prefill") in tp
        assert ("math", "prefill") in tp
        assert ("physics", "generation") not in tp  # target wasn't in rec2

    def test_channel_not_found_in_any_trace(self, tmp_path):
        _write_focus_trace(tmp_path / "test.jsonl", target_channel=42)
        out_json = tmp_path / "out.json"
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--traces", str(tmp_path / "*.jsonl"),
             "--channel", "999",
             "--out-json", str(out_json)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, f"stderr={result.stderr}"
        s = json.loads(out_json.read_text(encoding="utf-8"))
        assert s["n_records_with_channel"] == 0
        assert s["n_records_rank1"] == 0
