# GLM-5.2 Trace Artifacts

This directory holds MoE expert-routing traces for GLM-5.2, produced by the
Phase 1 tracer.  See `GLM52_TRACE_PLAN.md` for the full plan and acceptance
criteria, and `GLM52_SESSION_MEMORY.md` for the canonical session record.

## Layout

```
traces/
  glm52-coding-en-real-sample.jsonl           # real single-run sample (small)
  glm52-coding-en-real-sample.jsonl.meta.json # its metadata sidecar
  synth/                                       # synthetic traces from the smoke suite
    synth-<test_id>-<lang>.jsonl
    synth-<test_id>-<lang>.jsonl.meta.json
  smoke/<timestamp>/                           # live traced smoke runs (created by run_trace_task_suite.sh)
```

## Schema (v1)

Each `.jsonl` line is one ``(token, layer)`` routing event:

```json
{
  "schema_version": 1,
  "event": "moe_topk",
  "run_id": "...",
  "model": "GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest",
  "phase": "prefill",            // "prefill" | "generation"
  "token_index": 0,
  "layer": 3,
  "experts": [250, 185, 237, 218, 63, 50, 140, 172],
  "weights": [0.968, 0.291, ...],
  "router_entropy": 2.83175,
  "n_expert_used": 8,
  "n_expert": null,              // absent: no stable llama.cpp API to read total expert count
  "task_label": "coding",
  "language": "en",
  "script": "Latin",
  "prompt_family": "coding",
  "test_id": "coding_01_iterative_merge_sort"
}
```

The companion `<trace>.meta.json` sidecar holds run-level provenance: model path,
command line, prompt sha256, thinking mode, queue/backpressure counters, perf,
wall time, prompt/gen token counts.  It is written by the backend at exit and
tolerated-missing by the analyzer (so resumed/partial traces still analyze).

## Run lifecycle

1. **Backend writes JSONL only** — an async writer thread drains a bounded queue;
   the eval callback never performs file I/O (see `GLM52_TRACE_PLAN.md` →
   "Async logging requirement").
2. **Backend writes `.meta.json`** at exit with counters + provenance.
3. **Trace output is never silently overwritten** — the writer refuses an
   existing path; rotate or use a fresh `run_id` (the timestamped `--trace-out`
   defaults in `run_glm52_moe_trace.sh` make this automatic).

## Backpressure

Configurable via `--trace-backpressure {block|drop|sample}` (default `sample`):

| mode    | behavior under queue pressure                |
|---------|----------------------------------------------|
| block   | block briefly, then drop if still full       |
| drop    | drop immediately when the queue is full      |
| sample  | adaptively shed load under pressure (default)|

The metadata sidecar reports `records_written`, `records_dropped`, and
`records_sampled`.  Phase 1 traces the compact tensors only (`ffn_moe_topk`,
`ffn_moe_weights`) — GPU/Metal readback still synchronizes per layer, so use
`--trace-max-tokens` and `--trace-layers` to bound cost on 20k-token prompts.

## Producing traces

### Synthetic (no model load) — for analyzer/report demos and tests

```bash
cd "/Volumes/Data NVME/gguf2mlx"
source .venv/bin/activate
python3 scripts/tracing/make_synth_trace.py \
  --suite prompts/tracing/glm52_trace_smoke_suite.expanded.jsonl \
  --out-dir traces/synth
python3 scripts/tracing/analyze_moe_trace.py \
  --traces "traces/synth/*.jsonl" \
  --out-md  reports/glm52_moe_trace_report.md \
  --out-json reports/glm52_moe_trace_summary.json
```

### Live (real GLM-5.2 routing) — single prompt

```bash
bash scripts/tracing/run_glm52_moe_trace.sh
```

The wrapper defaults to the known-good mixed GGUF baseline and the patched
`llama-trace-moe` binary at
`/Users/spotted/projects/llama.cpp/build-metal/bin/llama-trace-moe`. Override via
`MODEL`, `CLI`, `OUT`, `PROMPT_FILE`, `TASK_LABEL`, `LANGUAGE`, `SCRIPT`,
`PROMPT_FAMILY`, `TEST_ID`, `NGL`, `CTX`, `N_PRED`, `PHASE`, `TRACE_LAYERS`,
`TRACE_MAX_TOKENS`, `BACKPRESSURE`.

### Live multilingual smoke suite

```bash
bash scripts/tracing/run_trace_task_suite.sh
```

This loops the expanded smoke suite across the 7 languages and 7 domains,
writes one trace per prompt under `traces/smoke/<timestamp>/`, then runs the
analyzer to produce `reports/glm52_moe_trace_report.md`.  Defaults to disabled
thinking + small `max_new_tokens` per the trace-plan thinking-budget policy.

To regenerate just the analyzer output over already-existing synthetic traces
(no live C++ runs):

```bash
SKIP_LIVE=1 TRACE_DIR=traces/synth bash scripts/tracing/run_trace_task_suite.sh
```

## Comparing runs / models

```bash
python3 scripts/tracing/compare_trace_reports.py \
  --label baseline   --summary reports/glm52_moe_trace_summary.json \
  --label reap37     --summary reports/reap37_moe_trace_summary.json \
  --out-md  reports/glm52_baseline_vs_reap37.md \
  --out-json reports/glm52_baseline_vs_reap37.json
```

Any label containing `reap` triggers the hardcoded INVALID-for-quality caveat
about REAP37 MLX indexer-compat (see `REAP37_EXPERIMENTS.md`).

## Limitations (Phase 1)

- **No `n_expert_total`** — there is no stable public llama.cpp API to read the
  total expert count; that field is omitted. The `n_expert_used` (experts
  selected per routing event) IS captured (real GLM-5.2 = 8).
- **DSA long-context retrieval (Phase 3) and activation summaries (Phase 4) are
  not implemented** — the analyzer reports this explicitly rather than silently
  omitting it.
- **Chat-template parity** — the tracer tokenizes `params.prompt` verbatim
  (mirrors `examples/eval-callback`); it does not apply the chat template
  itself. For `--jinja -cnv` parity with `llama-cli`, pass a pre-templated
  prompt via `-p`, or use the run scripts which rely on
  `common_init_from_params` honoring those flags for generation.
