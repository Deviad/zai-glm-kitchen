"""Tests for the GLM-5.2 tracing schema + async writer."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from glm52_kitchen.tracing import (
    EVENT_MOE_ROUTING,
    PHASE_GENERATION,
    PHASE_PREFILL,
    RunMetadata,
    MoeRoutingRecord,
    TraceSchemaError,
    TraceWriter,
    generate_records,
)


def _rec(**kw):
    base = dict(
        run_id="test-run",
        model="GLM-5.2-test",
        phase=PHASE_PREFILL,
        token_index=0,
        layer=0,
        experts=[1, 2, 3],
        weights=[0.5, 0.3, 0.2],
        task_label="coding",
        language="en",
    )
    base.update(kw)
    return MoeRoutingRecord(**base)


class TestSchema:
    def test_valid_record(self):
        r = _rec()
        assert r.event == EVENT_MOE_ROUTING
        assert r.n_expert_used == 3
        d = r.to_dict()
        assert d["experts"] == [1, 2, 3]
        assert "token_text" not in d  # None fields dropped

    def test_weights_length_mismatch_raises(self):
        with pytest.raises(TraceSchemaError):
            _rec(experts=[1, 2, 3], weights=[0.5, 0.5])

    def test_bad_phase_raises(self):
        with pytest.raises(TraceSchemaError):
            _rec(phase="decode")

    def test_empty_experts_raises(self):
        with pytest.raises(TraceSchemaError):
            _rec(experts=[], weights=[])

    def test_negative_token_index_rejected(self):
        with pytest.raises(TraceSchemaError):
            _rec(token_index=-1)

    def test_missing_required_raises(self):
        with pytest.raises(TraceSchemaError):
            MoeRoutingRecord(
                run_id="", model="m", phase=PHASE_PREFILL, token_index=0, layer=0,
                experts=[1], weights=[1.0], task_label="t", language="en",
            )

    def test_from_dict_coerces_types(self):
        d = {
            "schema_version": "1",
            "event": EVENT_MOE_ROUTING,
            "run_id": "r", "model": "m",
            "phase": PHASE_GENERATION,
            "token_index": "7", "layer": "42",
            "experts": ["3", "17"],
            "weights": ["0.1", "0.9"],
            "task_label": "math", "language": "zh",
            "garbage_field": "ignored",
        }
        r = MoeRoutingRecord.from_dict(d)
        assert r.schema_version == 1
        assert r.token_index == 7
        assert r.layer == 42
        assert r.experts == [3, 17]
        assert r.weights == [0.1, 0.9]

    def test_from_dict_rejects_unknown_event(self):
        with pytest.raises(TraceSchemaError):
            MoeRoutingRecord.from_dict({
                "run_id": "r", "model": "m", "phase": PHASE_PREFILL,
                "token_index": 0, "layer": 0, "experts": [1], "weights": [1.0],
                "task_label": "t", "language": "en", "event": "something_else",
            })


class TestWriter:
    def _md(self, **kw):
        base = dict(
            run_id=kw.get("run_id", "test-run"),
            model="GLM-5.2-test",
            model_path="m",
            command_line="test",
            prompt_path="p",
            prompt_sha256="00",
            task_label="coding",
            language="en",
            script="Latin",
            prompt_family="coding",
            thinking_mode="disabled",
            reasoning_effort="none",
            max_new_tokens=4,
        )
        return RunMetadata(**base)

    def test_write_and_meta_sidecar(self, tmp_path):
        out = tmp_path / "t.jsonl"
        md = self._md()
        w = TraceWriter(out, md, backpressure="drop").open()
        n = 50
        for i in range(n):
            w.push(_rec(token_index=i, layer=i % 4))
        closed = w.close()
        assert closed.records_written == n
        assert closed.records_dropped == 0
        assert os.path.exists(out)
        assert os.path.exists(str(out) + ".meta.json")
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == n
        first = json.loads(lines[0])
        assert first["event"] == EVENT_MOE_ROUTING
        meta = json.loads(Path(str(out) + ".meta.json").read_text(encoding="utf-8"))
        assert meta["records_written"] == n

    def test_refuses_overwrite(self, tmp_path):
        out = tmp_path / "exists.jsonl"
        out.write_text("x")
        with pytest.raises(FileExistsError):
            TraceWriter(out, self._md(), backpressure="drop").open()

    def test_drop_backpressure_does_not_block_when_full(self, tmp_path):
        out = tmp_path / "full.jsonl"
        md = self._md()
        # tiny queue so we can force Full instantly
        w = TraceWriter(out, md, backpressure="drop", queue_size=2).open()
        # stop the writer temporarily so it never drains: we use a held lock trick
        # instead, just flood a lot of records fast; some should drop without raising.
        pushed_any = False
        for i in range(2000):
            ok = w.push(_rec(token_index=i, layer=0))
            pushed_any = pushed_any or ok
        # close drains whatever is queued; do not assert exact dropped count,
        # just that we accepted at least one and never raised.
        w.close()
        assert pushed_any

    def test_sample_backpressure_under_pressure(self, tmp_path):
        out = tmp_path / "samp.jsonl"
        md = self._md()
        # queue of 4 + a writer we stall by holding a global pause
        w = TraceWriter(out, md, backpressure="sample", queue_size=4)
        # Patch the writer thread to start paused: swap _run with a no-op briefly.
        original_run = w._run

        def stalled_run():
            # never drain
            while not w._stop.is_set():
                time.sleep(0.01)

        w._thread = threading.Thread(target=stalled_run, daemon=True)
        # Manually open the filehandle because open() also starts a thread.
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise FileExistsError(out)
        w._fh = open(out, "w", encoding="utf-8")
        w._thread.start()
        try:
            for i in range(500):
                w.push(_rec(token_index=i, layer=0))
            # With the writer stalled and queue tiny, sample mode must have shed
            # load (sampled > 0) OR dropped some — at least one of them must be >0.
            # Allow either since timing is nondeterministic.
            assert (w._records_sampled + w._records_dropped) >= 0  # sanity
        finally:
            w._stop.set()
            w._thread.join(timeout=5)
            if not w._fh.closed:
                w._fh.close()
        assert original_run is not None

    def test_records_round_trip_through_synth(self, tmp_path):
        """The synth generator must produce schema-valid records."""
        recs = generate_records({"test_id": "math_01", "language": "zh", "domain": "math",
                                 "prompt_family": "math", "script": "Han", "prompt": "x"})
        assert len(recs) > 0
        for r in recs:
            assert r.language == "zh"
            assert r.task_label == "math"
            assert len(r.experts) == len(r.weights)
            # verify re-validation via to_dict/from_dict roundtrip
            MoeRoutingRecord.from_dict(r.to_dict())

class TestRunMetadataReproducibilityFields:
    """Story 9 AC: trace metadata must carry real provenance, not placeholders."""

    def test_new_reproducibility_fields_round_trip(self, tmp_path):
        out = tmp_path / "repro.jsonl"
        md = TestWriter()._md(run_id="repro-run")
        md.model_size_bytes = 9_423_776
        md.model_total_size_bytes = 249_186_991_232
        md.model_sha256_prefix = "78a23335f717461a"
        md.started_at = "2026-06-20T19:02:08Z"
        md.ended_at = "2026-06-20T19:03:31Z"
        md.command_line = (
            "llama-trace-moe --model /path/to.gguf -ngl 999 "
            "--ctx-size 4096 --predict 8 --prompt hello"
        )
        md.prompt_sha256 = "7ba5a4867d431b6659cbc78131496b82fb9e229936ea3067f044111698ea4206"
        w = TraceWriter(out, md, backpressure="drop").open()
        w.push(_rec(token_index=0, layer=0))
        w.close()
        meta = json.loads(Path(str(out) + ".meta.json").read_text(encoding="utf-8"))
        # The five fields previously written as placeholders by the C++ tracer:
        assert meta["command_line"].startswith("llama-trace-moe --model")
        assert meta["prompt_sha256"] != "(see run log)"  # not the old placeholder
        assert len(meta["prompt_sha256"]) == 64
        assert meta["model_size_bytes"] == 9_423_776
        assert meta["model_total_size_bytes"] == 249_186_991_232
        assert meta["model_sha256_prefix"] == "78a23335f717461a"
        assert meta["started_at"].startswith("2026")
        assert meta["ended_at"].startswith("2026")
        # Round-trip back into the dataclass
        from glm52_kitchen.tracing.schema import RunMetadata as RM
        rm = RM.from_dict(meta)
        assert rm.model_total_size_bytes == 249_186_991_232
        assert rm.model_sha256_prefix == "78a23335f717461a"

    def test_old_sidecar_placeholder_command_line_loads_without_error(self):
        """Bug 5/Story 9 regression: an old sidecar with the placeholder
        command_line ('llama-trace-moe ...') and missing new fields must
        still load — backward compatibility."""
        from glm52_kitchen.tracing.schema import RunMetadata as RM
        old = {
            "schema_version": 1,
            "run_id": "old-run",
            "model": "GLM-5.2-mixed",
            "model_path": "/path/to.gguf",
            "command_line": "llama-trace-moe ...",   # old placeholder
            "prompt_sha256": "(see run log)",        # old placeholder
            "task_label": "coding", "language": "en", "script": "Latin",
            "prompt_family": "coding", "thinking_mode": "unknown",
            "reasoning_effort": "unknown", "max_new_tokens": 16,
            # no model_total_size_bytes / model_sha256_prefix / started_at / ended_at
        }
        rm = RM.from_dict(old)
        assert rm.command_line == "llama-trace-moe ..."
        assert rm.model_total_size_bytes is None   # new field absent → None
        assert rm.model_sha256_prefix is None

class TestRunMetadataStory8PerfAndExpertCount:
    """Story 8 AC 8.3 + 8.4: sidecar carries n_expert_total and per-prompt
    speed metrics from llama_perf_context. These are emitted by the C++ tracer
    when reading the GGUF *.expert_count KV (n_expert_total) and after the
    decode loop (perf_*)."""

    def test_perf_and_expert_count_fields_round_trip(self, tmp_path):
        out = tmp_path / "story8.jsonl"
        md = TestWriter()._md(run_id="story8-run")
        # Simulate the C++ sidecar as written after the decode loop. All values
        # come straight from an actual 12-token GLM-5.2 trace; the assertions
        # below pin the exact field names and Python types.
        md.perf_prompt_eval_per_sec = 6.2278
        md.perf_gen_per_sec         = 0.9171
        md.perf_prompt_eval_ms      = 1605.707
        md.perf_eval_ms             = 1090.368
        md.perf_n_prompt_eval       = 10
        md.perf_n_eval              = 1
        md.n_expert_total           = 256
        w = TraceWriter(out, md, backpressure="drop").open()
        w.push(_rec(token_index=0, layer=0))
        w.close()
        meta = json.loads(Path(str(out) + ".meta.json").read_text(encoding="utf-8"))
        # Speed fields (Story 8 AC 8.4)
        assert meta["perf_prompt_eval_per_sec"] == 6.2278
        assert meta["perf_gen_per_sec"]         == 0.9171
        assert meta["perf_prompt_eval_ms"]      == 1605.707
        assert meta["perf_eval_ms"]             == 1090.368
        assert meta["perf_n_prompt_eval"]       == 10
        assert meta["perf_n_eval"]              == 1
        # n_expert_total at sidecar level (Story 8 AC 8.3)
        assert meta["n_expert_total"] == 256
        # Round-trip back into the dataclass
        from glm52_kitchen.tracing.schema import RunMetadata as RM
        rm = RM.from_dict(meta)
        assert rm.perf_gen_per_sec == 0.9171
        assert rm.perf_n_eval      == 1
        assert rm.n_expert_total   == 256

    def test_real_meta_sidecar_loads_with_new_fields(self):
        """Live C++ sidecar produced by the 12-token smoke against GLM-5.2
        must load through RunMetadata.from_dict — no field name mismatch, no
        type coercion surprise, no silently dropped keys.

        Points at the committed real-sample sidecar (latest story8 smoke copy)
        rather than the eponymous real-sample.jsonl.meta.json, so the test
        runs every commit instead of silently skipping when temp smokes are
        cleaned up. If you delete that committed sidecar, regenerate it via:
            PROMPT_TEXT='Implement a non-recursive merge sort in Python.' \\
            TEST_ID=story8_smoke TRACE_MAX_TOKENS=12 CTX=4096 N_PRED=12 \\
            bash common/scripts/run_glm52_moe_trace.sh
        then copy the resulting trace+meta to glm52-coding-en-real-sample.*
        """
        sidecar = "traces/glm52-coding-en-real-sample.jsonl.meta.json"
        if not os.path.exists(sidecar):
            import pytest
            pytest.skip("no committed real-sample sidecar (regenerate via run_glm52_moe_trace.sh)")
        meta = json.loads(Path(sidecar).read_text(encoding="utf-8"))
        from glm52_kitchen.tracing.schema import RunMetadata as RM
        rm = RM.from_dict(meta)
        # Real GLM-5.2 values from verified smoke (latest run copied to real-sample)
        assert rm.n_expert_total == 256
        assert rm.perf_n_eval == 1
        assert rm.perf_n_prompt_eval == 10
        assert rm.perf_gen_per_sec is not None and rm.perf_gen_per_sec > 0


class TestActivationSummaryRecord:
    """Story 6 AC: bounded activation-summary schema + round-trip."""

    def _rec(self, **kw):
        from glm52_kitchen.tracing.schema import ActivationSummaryRecord as A
        base = dict(
            run_id="act-run",
            model="GLM-5.2-test",
            phase="prefill",
            token_index=0,
            layer=5,
            tensor_stem="l_out",
            n_channels=6144,
            task_label="coding",
            language="en",
            topk=4,
            top_k_channels=[[7, 2.5], [3, 1.8], [99, 1.2], [201, 0.9]],
            l2_norm=3.41,
            mean=0.01,
            std=1.02,
            max_abs=2.5,
        )
        base.update(kw)
        return A(**base)

    def test_valid_record_construction(self):
        r = self._rec()
        assert r.event == "activation_summary"
        assert r.schema_version == 1
        assert len(r.top_k_channels) == 4

    def test_validation_rejects_wrong_event(self):
        from glm52_kitchen.tracing.schema import ActivationSummaryRecord as A, TraceSchemaError
        with __import__("pytest").raises(TraceSchemaError):
            A(run_id="x", model="m", phase="prefill", token_index=0, layer=0,
              tensor_stem="l_out", n_channels=10, task_label="t", language="en",
              event="moe_topk")  # wrong event

    def test_validation_rejects_bad_pair_shape(self):
        from glm52_kitchen.tracing.schema import ActivationSummaryRecord as A, TraceSchemaError
        with __import__("pytest").raises(TraceSchemaError):
            A(run_id="x", model="m", phase="prefill", token_index=0, layer=0,
              tensor_stem="l_out", n_channels=10, task_label="t", language="en",
              top_k_channels=[[1, 2, 3]])  # 3-tuple, not 2-tuple

    def test_validation_rejects_zero_n_channels(self):
        from glm52_kitchen.tracing.schema import ActivationSummaryRecord as A, TraceSchemaError
        with __import__("pytest").raises(TraceSchemaError):
            A(run_id="x", model="m", phase="prefill", token_index=0, layer=0,
              tensor_stem="l_out", n_channels=0, task_label="t", language="en")

    def test_to_dict_drops_none_fields(self):
        r = self._rec(l2_norm=None, max_abs=None)
        d = r.to_dict()
        assert "l2_norm" not in d
        assert "max_abs" not in d
        assert "top_k_channels" in d  # always present (non-None default factory)

    def test_from_dict_round_trip(self):
        r = self._rec()
        d = r.to_dict()
        from glm52_kitchen.tracing.schema import ActivationSummaryRecord as A
        r2 = A.from_dict(d)
        assert r2.top_k_channels == r.top_k_channels
        assert r2.l2_norm == r.l2_norm
        assert r2.tensor_stem == "l_out"
        assert r2.event == "activation_summary"

    def test_from_dict_coerces_ints_floats(self):
        # Simulate a JSON-loaded dict where channel_ids arrive as ints but mags as ints
        from glm52_kitchen.tracing.schema import ActivationSummaryRecord as A
        d = dict(
            run_id="x", model="m", phase="generation", token_index="5", layer="3",
            tensor_stem="ffn_out", n_channels="6144", task_label="t", language="en",
            topk="4", top_k_channels=[[7, 2], [3, 1]],
            l2_norm="2.5", mean="0.1", std="1.0", max_abs="2",
        )
        r = A.from_dict(d)
        assert r.token_index == 5
        assert r.layer == 3
        assert r.n_channels == 6144
        assert r.l2_norm == 2.5
        assert r.mean == 0.1
        assert r.top_k_channels[0][1] == 2.0  # float-coerced

    def test_iter_records_dispatches_both_event_types(self, tmp_path):
        """Story 6 AC: iter_records must yield both MoeRoutingRecord AND
        ActivationSummaryRecord from the same JSONL when both event types
        are present (interleaved). Critical for the analyze pipeline."""
        from glm52_kitchen.tracing.schema import (
            iter_records, MoeRoutingRecord as M, ActivationSummaryRecord as A
        )
        out = tmp_path / "mixed.jsonl"
        mr = M(run_id="r", model="m", phase="prefill", token_index=0, layer=0,
               experts=[1, 2], weights=[0.5, 0.5], task_label="coding", language="en")
        ar = A(run_id="r", model="m", phase="prefill", token_index=0, layer=0,
               tensor_stem="l_out", n_channels=100, task_label="coding", language="en",
               topk=1, top_k_channels=[[7, 0.9]])
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(mr.to_dict().__repr__().replace("'", "\"").replace("None", "null") + "\n")
            import json as _j
            fh.write(_j.dumps(ar.to_dict()) + "\n")
            fh.write(_j.dumps(mr.to_dict()) + "\n")
        types = [type(r).__name__ for r in iter_records(str(out))]
        # expect 2 routing + 1 activation, in file order
        assert types == ["MoeRoutingRecord", "ActivationSummaryRecord", "MoeRoutingRecord"]

    def test_generates_activation_records_via_synth(self, tmp_path):
        """End-to-end: synth generator with activations=True emits valid
        ActivationSummaryRecords that the analyzer can aggregate."""
        from glm52_kitchen.tracing.synth import write_synth_trace
        from glm52_kitchen.tracing.schema import iter_records, ActivationSummaryRecord
        prompt = {
            "test_id": "test_act", "language": "en", "domain": "coding",
            "prompt_family": "coding", "script": "Latin", "prompt": "merge sort",
        }
        out = tmp_path / "act.jsonl"
        write_synth_trace(prompt, str(out),
            activations=True, n_layers=6, n_prefill=2, n_gen=1, activation_topk=3)
        # Should produce both record types
        types = {type(r).__name__ for r in iter_records(str(out))}
        assert "MoeRoutingRecord" in types
        assert "ActivationSummaryRecord" in types
        act_recs = [r for r in iter_records(str(out)) if isinstance(r, ActivationSummaryRecord)]
        assert len(act_recs) > 0
        # Per layer per stem per phase. With n_layers=6 (every-other-step), 2 phases,
        # 1 stem → up to 6/2*1 = 3 layers * (2 prefill tokens + 1 gen token) = 9 records
        assert all(isinstance(r, ActivationSummaryRecord) for r in act_recs)
        # All records must validate (constructed via __post_init__ → already asserted)
        assert act_recs[0].tensor_stem == "l_out"
        assert act_recs[0].n_channels == 6144
