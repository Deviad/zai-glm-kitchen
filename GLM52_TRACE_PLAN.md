# GLM-5.2 Debugging / Activation Trace Plan

Date: 2026-06-20

## Phase 1 — Implemented

Phase 1 (MoE expert routing tracer) is **implemented** and verified end-to-end
against the known-good mixed GGUF baseline. Artifacts:

```text
src/gguf2mlx/tracing/          # schema + async writer + analyzer + comparator + synth
tests/test_tracing_schema_writer.py
tests/test_tracing_analyze.py
scripts/tracing/run_glm52_moe_trace.sh       # single-prompt live traced run
scripts/tracing/run_trace_task_suite.sh      # multilingual smoke-suite traced run
scripts/tracing/analyze_moe_trace.py         # JSONL -> markdown report + summary
scripts/tracing/compare_trace_reports.py     # side-by-side model/run comparison
scripts/tracing/make_synth_trace.py          # synthetic traces for tests/demo
traces/README.md                              # schema, paths, run lifecycle
~/projects/llama.cpp/examples/trace-moe/      # C++ backend (cb_eval on ffn_moe_topk/weights)
```

Verified real-data run on the mixed GLM-5.2 model produced 32 schema-valid
routing records (8 experts/event, softmax weights + entropy), 0 dropped, 0
sampled, with a complete `.meta.json` sidecar, and the analyzer produced a
report from it. 20 unit tests + the 54 existing tests all pass (74 total).

Key implementation notes:

- The C++ backend hooks the ggml backend eval callback (`cb_eval`), filters by
  tensor name (`ffn_moe_topk-N` / `ffn_moe_weights-N`), pairs them per
  `(token, layer)`, reads host data via `ggml_backend_tensor_get`, and pushes
  compact JSONL records to a bounded queue drained by a writer thread.
- Backpressure `block|drop|sample` is implemented; default `sample`.
- The analyzer is backend-agnostic (also consumes the synthetic generator),
  producing: top experts by task/language/layer, task & language Jaccard overlap,
  expert specialization, router entropy, prefill-vs-generation, and tokenization
  stats per language from metadata.
- The comparator carries the hardcoded caveat that REAP37 MLX indexer-compat
  traces are INVALID for quality comparison.

Phases 2b (DSA long-context) / 3 / 4 / 5 below remain planned, not implemented.

## Objective

Build a reproducible tracing workflow for GLM-5.2 that shows which model components activate during different task types, especially:

1. coding tasks
2. reading scientific / technical documentation
3. explaining or summarizing technical material
4. defensive cybersecurity analysis and security engineering tasks
5. multilingual behavior in English, Italian, Chinese, Spanish, French, German, and Portuguese

Start with low-overhead MoE expert routing traces. Add deeper neuron / activation tracing only after the first layer of evidence is useful.

## Guiding principles

- Tracing must be opt-in and disabled by default.
- Existing known-good GLM-5.2 GGUF baseline must not be overwritten.
- Trace output must be structured and analyzable, not just logs.
- Start with MoE experts because GLM-5.2 is a sparse MoE and expert routing is the clearest first proxy for “which neurons/components activated”.
- Avoid dumping full activations for 20k-token prompts unless sampled or bounded; raw full activation traces will become huge.
- Every experiment must be reproducible through scripts.
- Fast smoke traces should limit or disable visible thinking to reduce runtime and avoid mixing answer behavior with long reasoning behavior. Separate reasoning-quality runs can enable thinking on a smaller prompt subset.

## Thinking-budget policy

For the 20-test × 7-language smoke suite, limiting thinking is recommended and low risk if the purpose is routing/activation comparison rather than final benchmark quality.

Recommended modes:

```text
fast_trace_smoke:
  thinking: disabled or minimal
  max_new_tokens: small, e.g. 128-384 depending on task
  temperature: 0 or low
  purpose: routing/activation comparability and speed

reasoning_quality_subset:
  thinking: enabled / higher effort
  max_new_tokens: larger
  prompt subset: representative hard math, physics, coding, and science explanation tasks
  purpose: measure whether thinking changes expert routing and answer quality
```

Why disable/limit thinking for smoke tracing:

- generated thinking tokens dominate runtime and trace volume
- long reasoning can make all languages/tasks look more similar because the model enters a generic reasoning mode
- smoke tests are mainly for relative expert routing by language/domain
- bounded outputs make cross-language comparisons cleaner

Risks:

- some math, physics, chemistry, and debugging prompts may be less accurate without thinking
- expert routing with thinking disabled may not represent full reasoning behavior
- if the research question is “how does GLM reason?”, then thinking should be enabled for a targeted subset

Therefore each trace run must record thinking mode in metadata:

```json
{
  "thinking_mode": "disabled",
  "reasoning_effort": "none",
  "max_new_tokens": 256
}
```

## Proposed scope

### Phase 1 — MoE expert routing tracer

Capture per token / per layer:

- selected expert IDs
- selected expert weights
- router confidence / entropy
- token position
- layer number
- prompt/task label
- decode phase: prefill vs generation

Relevant llama.cpp graph tensors already exist:

```text
ffn_moe_logits
ffn_moe_probs
ffn_moe_topk
ffn_moe_weights
```

### Phase 2 — Analysis and reports

Aggregate traces into:

- expert usage heatmaps by layer and task
- top experts for coding vs scientific reading vs explaining vs defensive cybersecurity
- top experts by language: English, Italian, Chinese, Spanish, French, German, Portuguese
- overlap / specialization scores
- language-vs-task disentanglement scores, using parallel prompts translated across languages
- router entropy by task, language, and layer
- per-phase comparisons: prefill vs generation

### Phase 2b — Language-conditioned routing trace

There may not be explicit language-specific tensors such as an “Italian tensor” or “Chinese tensor”. Instead, language behavior should be studied by comparing activation and routing patterns while controlling for task content.

Capture and report:

- language label: `en`, `it`, `zh`, `es`, `fr`, `de`, `pt`
- script: Latin vs Han / Chinese
- prompt family: coding, science-reading, explanation, translation, summarization
- tokenization stats: token count, character count, tokens per character/word where applicable
- per-layer MoE expert usage by language
- per-language router entropy / confidence
- experts that activate consistently for one language across multiple task families
- experts that activate for the same task regardless of language
- code-switching behavior when prompts mix two languages

Initial language suite should use semantically parallel prompts: the same coding request, scientific paragraph, and explanation request translated into each tracked language.

### Phase 3 — DSA / long-context retrieval trace

For GLM-5.2 DSA / IndexShare behavior, capture:

- selected long-context positions from DSA top-k indexer
- distance distribution of selected tokens
- whether coding and scientific-document prompts retrieve different context regions

### Phase 4 — Bounded neuron / activation summaries

Only after MoE traces are useful, add sampled activation summaries:

- top activated MLP channels
- activation norms
- routed expert vs shared expert contribution
- per-layer activation sparsity

Do not dump every full activation by default.

### Phase 5 — Optional ablation experiments

Once candidate task-specialized experts/channels are identified:

- disable or downscale selected experts
- compare coding/science/explanation quality
- measure speed and quality impact

This converts correlation into stronger causal evidence.

## Planned artifacts

```text
scripts/tracing/run_glm52_moe_trace.sh
scripts/tracing/run_trace_task_suite.sh
scripts/tracing/analyze_moe_trace.py
scripts/tracing/compare_trace_reports.py
traces/README.md
reports/glm52_moe_trace_report.md
```

Potential llama.cpp changes:

```text
/Users/spotted/projects/llama.cpp/examples/trace-moe/...
```

or a patch to `llama-cli` with a flag such as:

```text
--trace-moe path/to/trace.jsonl
--trace-task-label coding
--trace-max-tokens N
--trace-layers 0,1,2,6,10,...
```

## Trace output decision

Output redirection alone is not sufficient for the core trace data. It is useful for human-readable run logs, command output, warnings, and performance lines, but expert IDs, weights, entropy, token positions, and activation summaries should be emitted by the tracing function itself to a structured file.

Recommended split:

```text
stdout/stderr redirection -> human run log, generated text, llama.cpp timing lines
trace JSONL file          -> machine-readable routing / activation events
metadata JSON file        -> run config, prompt hash, model path, command line
```

Reasons:

- stdout can interleave generated text, progress output, warnings, and trace lines.
- parsing human logs is brittle.
- structured JSONL supports deterministic analysis and comparison.
- trace records need schema version, run ID, task label, phase, token index, layer, and event type.
- the tracer can buffer writes and avoid excessive `printf` overhead.

Default behavior should create a new timestamped trace file or fail if the target exists. Append mode should be explicit and include a stable `run_id`, so resumed/incremental traces remain distinguishable.

## Async logging requirement

Trace writing should run on a separate writer thread so normal model execution is disturbed as little as possible.

Recommended design:

```text
inference / eval thread:
  - observes selected tensors
  - extracts only the small values needed for the enabled trace mode
  - builds compact trace records
  - pushes records into a bounded queue
  - never performs file I/O directly

writer thread:
  - drains the queue
  - batches records
  - writes JSONL and flushes periodically
  - writes final metadata and close markers
```

The inference thread must not block on disk I/O. If the queue is full, behavior should be configurable:

```text
--trace-backpressure block   # exact trace, may slow inference
--trace-backpressure drop    # preserve inference speed, count dropped records
--trace-backpressure sample  # adaptive sampling under pressure
```

Default should prefer preserving inference speed:

```text
--trace-backpressure sample
```

Caveat: a separate writer thread removes file I/O overhead, but it does not eliminate the cost of reading tensor values from GPU/Metal memory. Tensor readback can force synchronization. Therefore Phase 1 should trace only compact tensors such as `ffn_moe_topk` and `ffn_moe_weights`, and should support layer/token sampling.

Acceptance requirements for async logging:

- File writes are never performed from the model eval callback/inference hot path.
- Trace events are queued and written by a dedicated writer thread.
- Queue size and dropped/sampled record counts are reported in metadata.
- The tracer supports a no-block mode that prioritizes model performance.
- Benchmarks compare tracing disabled vs tracing enabled to quantify overhead.

## Trace output schema draft

JSONL, one record per traced tensor/token/layer event:

```json
{
  "schema_version": 1,
  "model": "GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest",
  "task_label": "coding",
  "language": "en",
  "script": "Latin",
  "prompt_family": "coding",
  "run_id": "2026-06-20-glm52-coding-001",
  "phase": "prefill",
  "token_index": 123,
  "token_text": "def",
  "layer": 42,
  "event": "moe_topk",
  "experts": [17, 92, 104, 3, 55, 201, 9, 77],
  "weights": [0.21, 0.17, 0.14, 0.12, 0.11, 0.09, 0.08, 0.08],
  "router_entropy": 1.92
}
```

## User stories and acceptance criteria

### Story 1 — Model experimenter traces expert routing

**As a model experimenter, I need to trace which MoE experts GLM-5.2 selects for each token and layer, so that I can compare how coding, scientific reading, and explanation tasks use the sparse model.**

Acceptance criteria:

- ✅ Given a GLM-5.2 GGUF model and a prompt, when I run the MoE trace script, then a JSONL trace file is created. (Verified on real GLM-5.2 mixed GGUF, multiple runs.)
- ✅ The trace includes task label, token index, layer, selected expert IDs, and expert weights. (Schema fields `task_label`, `token_index`, `layer`, `experts`, `weights`.)
- ✅ Tracing is opt-in and normal inference works unchanged when tracing is disabled. (`--trace-out` is required; without it, the binary refuses to start. No `--trace-*` flags → standard llama.cpp inference.)
- ✅ The trace script works with the current known-good mixed GGUF baseline. (Verified against 232GB GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest.)
- ✅ The trace does not overwrite existing baseline output files. (Output paths default to timestamped: `traces/glm52-<task>-<lang>-<test_id>-<YYYYMMDD-HHMMSS>.jsonl`; the writer refuses to overwrite. Baseline scripts live in `scripts/baselines/` and output to separate `outputs/` paths.)

### Story 2 — Researcher compares task-specific expert usage

**As a researcher, I need aggregate reports comparing expert usage across task categories, so that I can identify experts or layers that appear specialized for coding, scientific documentation, or explanation.**

Acceptance criteria:

- ✅ Given trace files from at least three task categories, when I run the analyzer, then it produces a markdown report and machine-readable summary. (Verified on 7-task × 7-language 161-prompt study.)
- ✅ The report lists top experts by task and layer. (`## Top experts by task and layer` section.)
- ✅ The report includes overlap metrics between task categories. (`## Task overlap (Jaccard of pooled top-N experts)` section.)
- ✅ The report includes router entropy / confidence summaries. (`## Router entropy` section.)
- ✅ The report clearly distinguishes prefill from generation behavior. (Prefill/generation are tagged per record; 161-prompt study confirmed they are fully disjoint at Jaccard 0.0.)

### Story 3 — Performance-conscious user bounds trace overhead

**As a local model user, I need tracing to be bounded and configurable, so that I can collect useful evidence without making 20k-token runs impractically slow or producing massive trace files.**

Acceptance criteria:

- ✅ The trace runner supports maximum traced tokens. (`--trace-max-tokens N`; env `TRACE_MAX_TOKENS` in wrapper.)
- ✅ The trace runner supports selecting layers or layer ranges. (`--trace-layers "0,1,2,6"` or `--trace-layers "0..78"`; env `TRACE_LAYERS` in wrapper.)
- ✅ The trace runner supports prefill-only, generation-only, or both. (`--trace-phase prefill|generation|both`; env `TRACE_PHASE` in wrapper.)
- ✅ The output file size is reported at the end of the run. (`run_one_prompt` log line includes records written/dropped/sampled + wall; meta.json sidecar includes `records_written`.)
- ✅ Trace file writing runs on a separate writer thread and is not performed in the inference hot path. (`TraceWriter` owns a dedicated `std::thread` that drains a bounded queue; the eval callback only builds compact records + `push()`, never writes files.)
- ✅ Queue size, dropped record count, and sampled record count are reported. (`queue_size`, `records_dropped`, `records_sampled` in meta.json + run_one_prompt stdout.)
- ✅ A 20k-token retrieval prompt can be traced in a reduced/sampled mode without exhausting disk space. (`--trace-backpressure sample --trace-max-tokens 32` on a 20k prompt produces ~32 × n_layers records, << 1 MB. The baseline `scripts/baselines/glm52_longctx_retrieval_baseline.sh` uses the same 20k prompt; the tracer can run on it.)

### Story 4 — Developer validates trace correctness

**As a developer implementing the tracer, I need tests that prove tensor names, shapes, and extracted values are correct, so that trace results are trustworthy.**

Acceptance criteria:

- ✅ Unit or integration tests verify that `ffn_moe_topk` extraction returns integer expert IDs. (`tests/test_tracing_schema_writer.py` `TestSchema.test_from_dict_coerces_types` asserts `experts == [1, 2, 3]` from raw JSON.)
- ✅ Tests verify that `ffn_moe_weights` extraction returns the expected shape: `[n_expert_used, n_tokens]` or equivalent normalized output. (`test_weights_length_mismatch_raises` + `test_empty_experts_raises` enforce `len(experts) == len(weights)`; the C++ tracer uses topk's `n_used` for both tensors — Bug 2 fix.)
- ✅ Tests run on a tiny or synthetic MoE model if possible. (Schema/writer/analyze tests run on synthetic JSONL records — no MoE model load needed. `make_synth_trace.py` produces 161 synthetic traces for end-to-end pipeline verification.)
- ✅ If a full GLM-5.2 run is required, it is marked as an optional/manual integration test. (No test in the pytest suite loads GLM-5.2; the real-model smoke tests are run manually via `scripts/tracing/run_glm52_moe_trace.sh` and `run_trace_suite_batched.sh`, gated on the 26 GB GGUF being present.)
- ✅ The tracer fails with a clear error if an expected tensor is missing. (`trace_cb_eval` verifies `ggml_nelements(topk) > 0` and matches `n_used` from topk to weights; mismatches are skipped. `--trace-out` missing → exits with code 2. Both `-p` and `-f` missing → exits with code 2 with a clear error message.)

### Story 5 — Documentation reader traces long-context retrieval behavior

**As someone studying GLM-5.2 long-context behavior, I need to trace which earlier token positions the DSA indexer retrieves, so that I can see whether the model is actually reading relevant documentation sections.**

Acceptance criteria:

- ✅ The DSA trace records selected context positions or position buckets for traced layers. (DONE 2026-06-20 — RE-SCOPED to MLA retrieval patterns after the DSA forward-path patch was empirically rejected. The C++ tracer captures `q_nope_absorbed` (absorbed query — what each gen-step token asks for) and `kv_cmpr` (lora-compressed KV — what each prefill token offers) per (token, layer) via existing `--trace-activations q_nope_absorbed,kv_cmpr`. New analyzer module `src/gguf2mlx/tracing/retrieval.py` computes per-(query, layer) top-N retrieved prefill positions via signed top-K channel overlap (Σ over shared channels of q·c × k·c). Real-model long-ctx run on the BLUE-FALCON-48217 retrieval prompt (18,745 prefill tokens, 24 gen tokens, stride=8, topk=20) → 240 (query, layer) pairs scored, distance-bucketed, sentinel-overlapped via `--sentinel-position-range`. Full pipeline: `analyze_moe_trace.py --retrieval-stems q_nope_absorbed,kv_cmpr --retrieval-topn 10 --sentinel-position-range 50,60`. See `GLM52_SESSION_MEMORY.md` → "Story 5 re-scoped to MLA retrieval patterns — IMPLEMENTED" for full design + results.)
- ✅ The analyzer reports distance buckets such as recent, medium-context, and far-context retrieval. (DONE 2026-06-20. Analyzer module `src/gguf2mlx/tracing/retrieval.py` calls `distance_bucket(query_pos, retrieved_pos, prompt_len)` for every retrieved position, mapping to buckets: recent (≤5% of prompt_len or ≤64 tokens), medium (5%-30%), far (30%-70%), very_far (>70% — start of prompt), future (defensive: retrieved_pos ≥ query_pos). Summary dict carries `bucket_counts` + `bucket_fractions`. Report renders the '### Distance buckets' table. Real-model long-ctx run: 47 recent (2.0%), 2353 very_far (98.0%) — strongly skewed to 'start of prompt' which matches the BLUE-FALCON task's 'sentinel from near the beginning' instruction.)
- ✅ The report can show whether the sentinel section in the long-context retrieval prompt was selected. (DONE 2026-06-20. Analyzer accepts `--sentinel-position-range START,END` (inclusive) and counts `sentinel_hits` / `sentinel_total` — fraction of (query, layer) pairs whose top-N retrieved positions included ≥1 position in the sentinel range. Summary dict carries `sentinel_hits`, `sentinel_total`, `sentinel_hit_rate`. Report renders the '### Sentinel section retrieval' subsection. Real-model long-ctx BLUE-FALCON run with sentinel_range=(50,60) → 17.5% hit rate (42 / 240 (query, layer) pairs). Chance baseline: 1 - (1 - 11/18745)^10 ≈ 0.59% — the model retrieves sentinel token positions ~30x more often than chance. Direct hit example: layer 56 query 18768's TOP-1 retrieved position = 57 (inside the sentinel range), score 7.770, 5 shared channels.)
- ✅ DSA tracing can be disabled independently from MoE tracing. (DONE 2026-06-20 via the re-scoped design: the MLA retrieval analyzer is GATED by `--retrieval-stems q,k` on `analyze_moe_trace.py` (off by default — no overhead when not requested) AND by `--trace-activations q_nope_absorbed,kv_cmpr` on the C++ tracer (off by default — option `TraceConfig::activation_stems.empty()` check returns early, no activation records emitted). Both gates are independent from the existing `--trace-moe` / `--trace-max-tokens` flags. The C++ activation path also gates per-layer via `--trace-activation-stride N` (only every Nth layer emits activation records) and per-token via `--trace-max-tokens` (shared with MoE routing). Verified: real-model smoke 1 (no flags) → 0 activation records, only MoE routing records; real-model smoke 2 (with `--trace-activations q_nope_absorbed,kv_cmpr --trace-activation-topk 5`) → emits both record types.)
- ✅ If DSA tracing is not implemented in Phase 1, the report explicitly marks it as unavailable rather than silently omitting it. (Every analyzer markdown report ends with: _"DSA long-context retrieval tracing (Phase 3) is **not yet implemented** in this report; see `GLM52_TRACE_PLAN.md`. Activation summaries (Phase 4) are disabled by default and require explicit flags."_)

### Story 6 — Interpretability analyst inspects activation summaries

**As an interpretability analyst, I need bounded summaries of MLP/neuron activations, so that I can move beyond expert IDs and inspect which internal channels are strongest without dumping every activation.**

Acceptance criteria:

- ✅ Activation tracing supports top-k channels per selected tensor. (Story 6 AC 6.1: DONE 2026-06-20. C++ tracer takes `--trace-activations <stem1,stem2,...>` to opt into named intermediate tensors (e.g. `l_out`, `kqv_out`, `ffn_out`, `ffn_moe_out`, `ffn_swiglu`, `attn_norm`, `ffn_norm`). `trace_cb_eval` intercepts matching `<stem>-N` tensors via `is_activation_tensor()` + extract_layer(). Per-token top-K channels computed via O(n_channels log topk) min-heap — NOT an O(N²) partial_sort, which would dominate cost on 6144-channel prefill. Default topk=10, configurable via `--trace-activation-topk`. Pushes `render_activation_record()` JSONL rows with `top_k_channels: [[channel_idx, magnitude], ...]` sorted by |magnitude| desc; magnitude is signed so the analyzer can distinguish excitatory vs inhibitory channels. Verified on real GLM-5.2: `--trace-activations l_out --trace-activation-topk 5` produced real top-K channels per layer/token with sensible magnitudes ~0.05–0.21.)
- ✅ Activation tracing supports norm/stat summaries per layer. (Story 6 AC 6.2: DONE 2026-06-20. Per (token, tensor) summary: `l2_norm`, `mean`, `std`, `max_abs` computed in a single forward pass via sum/sumsq/max_abs accumulators (var numerically clamped to >=0 for robustness). Sidecar adds `activation_stems`, `activation_topk`, `activation_stride` so the analyzer detects optimization choice. Analyzer `## Bounded activation summaries` table renders per (task, layer, tensor_stem) row with mean L2 / mean mean / mean std / mean max_abs across per-token values.)
- ✅ Activation tracing can be sampled by token and layer. (Story 6 AC 6.3: DONE 2026-06-20. Two sampling axes: (1) layer-axis via `--trace-activation-stride N` (default 2 → emits only for every Nth layer; bounds prefill JSONL volume); (2) token-axis via already-existing `--trace-max-tokens N` (budget shared with MoE routing events per phase). Combined: on a 20k-token prefill through 79 layers with stride 4 → prefill produces ≤ (79/4) × min(max_tokens, 20k) activation records — bounded.)
- ✅ The trace schema distinguishes expert routing events from activation-summary events. (Story 6 AC 6.4: DONE 2026-06-20. `EVENT_MOE_ROUTING = "moe_topk"`; `EVENT_ACTIVATION_SUMMARY = "activation_summary"`. `iter_records()` yields `MoeRoutingRecord | ActivationSummaryRecord` union by dispatching on `event` field. `aggregate()` dispatches by record type: routing events go through `by_task_layer`/`by_lang_layer`/entropy aggregation; activation-summary events go to `agg.activation_summaries` for separation-concern-clean analysis. Both event types coexist in the same JSONL.)
- ✅ Full activation dumps are disabled by default and require an explicit flag. (Story 6 AC 6.5: DONE 2026-06-20. `trace_cb_eval` returns early when `st.activation_stems.empty()` — no activation records ever emitted without `--trace-activations`. Default behavior preserved for users who only want routing traces. `DEFAULT_ACTIVATION_STEMS = ("l_out",)` documents the single-stem default when the flag is passed without args.)

### Story 7 — Multilingual user traces language-conditioned behavior

**As a multilingual user of English, Italian, Chinese, Spanish, French, German, and Portuguese, I need traces grouped by language as well as by task, so that I can see whether GLM-5.2 uses different experts, layers, or activation patterns when I work in each language.**

Acceptance criteria:

- ✅ Trace metadata includes a language label for each run: `en`, `it`, `zh`, `es`, `fr`, `de`, or `pt`.
- ✅ Trace metadata includes prompt family, so language effects can be separated from task effects.
- ✅ The prompt suite includes semantically parallel prompts across all seven languages.
- ✅ The analyzer reports top experts by language and layer.
- ✅ The analyzer reports language overlap/specialization metrics.
- ✅ The analyzer reports tokenization statistics per language, including token count and tokens per character or word where applicable. (was unmet until Bug 5: `RunMetadata.prompt_path` required → sidecars silently failed to load → `tokenization_stats_per_language` empty. Fixed 2026-06-20. Mean prompt tokens by lang: en=29.0, zh=28.7, de=43.3 at N=161.)
- ✅ Reports distinguish language-specific patterns from task-specific patterns using same-task translated prompts. (Confirmed via the scaled 161-prompt study: the apparent zh/non-zh isolation at 49 prompts was a sampling artifact; multiple variants per cell are required to separate language from prompt-idiosyncratic routing.)
- ✅ Code-switching prompts can be labeled with multiple languages, such as `en+it` or `en+zh`. (DONE 2026-06-20: `prompts/tracing/glm52_code_switch_suite.expanded.jsonl` — 6 language pairs × 3 domains + 1 triple-language (en+zh+es) = 16 prompts. Analyzer treats `language` as opaque string, so multi-segment labels aggregate naturally. Findings: code-switch entropy = midpoint of component langs (-0.015 to +0.017 bits); code-switch top-10 = partial union (5-7 experts shared with each of the two langs at layer 10). 16-prompt N caveat noted; per-cell claims require ≥3 variants/cell.)

### Story 8 — Quantization/pruning evaluator compares baselines

**As a quantization and pruning evaluator, I need to compare expert-routing patterns between the current GGUF baseline and future pruned/REAP variants, so that I can see whether compression changes the model’s internal behavior.**

Acceptance criteria:

- ✅ Analyzer accepts multiple model/run labels. (`compare_trace_reports.py` takes `--label A --summary A.json --label B --summary B.json`; verified with baseline+reap37-compat labels.)
- ✅ Reports include side-by-side expert usage distributions. (`## Side-by-side top experts` section + per-label expert frequency tables.)
- ✅ Reports flag missing experts or changed expert counts. (DONE 2026-06-20: C++ tracer now reads `*.expert_count` from GGUF KV via `gguf_init_from_file` and populates `n_expert_total` (256 for GLM-5.2). Each routing record carries `"n_expert":256`; the sidecar also carries `n_expert_total`. `compare_trace_reports.py` was already prepared to compute set-diff across labels' expert unions ("Missing experts" section) and label-expert-count changes ("Expert-count changes" section, suppressed when both labels match — correct). Verified via smoke: 24 records each with `"n_expert":256`; compare against real+real-copy synth shows 0 missing experts, no expert-count changes — correct.)
- ✅ Reports include speed metrics if available from run logs. (DONE 2026-06-20: `run_one_prompt` now calls `llama_perf_context_reset(ctx)` at start of each prompt + `llama_perf_context(ctx)` at end to capture per-prompt `t_p_eval_ms`/`t_eval_ms`/`n_p_eval`/`n_eval`. Computes `perf_prompt_eval_per_sec` and `perf_gen_per_sec` (tok/s = tokens * 1000 / ms). Sidecar also emits `perf_prompt_eval_ms`/`perf_eval_ms`/`perf_n_prompt_eval`/`perf_n_eval`. Verified: smoke produced `perf_gen_per_sec=0.9171` (1 token/1.090s), `perf_n_eval=1`, `perf_n_prompt_eval=10`. Analyzer provenance block renders `Speed: **0.92 gen tok/s**, 6.23 prefill tok/s`; compare report's Speed section renders `0.92` mean gen/s per label.)
- ✅ The current REAP37 MLX compat result is marked invalid for quality comparison unless proper IndexShare support is implemented. (Hardcoded caveat in `compare_trace_reports.py`: _"REAP37 MLX indexer-compat traces are marked INVALID for quality comparison: stock mlx-lm lacks IndexShare support; see REAP37_EXPERIMENTS.md."_)

### Story 9 — Reproducibility maintainer preserves experiment context

**As a reproducibility maintainer, I need every trace run to save metadata, command lines, model paths, and prompt hashes, so that future runs can be repeated and compared.**

Acceptance criteria:

- ✅ Each trace output includes model path, model hash or size, prompt path, prompt hash, command line, date, llama.cpp build path, thinking mode, reasoning effort, and max output tokens. (DONE 2026-06-20: the C++ meta.json sidecar now carries all of: `model_path`, `model_size_bytes`, `model_total_size_bytes` (232 GB across 9 shards), `model_sha256_prefix` (first 1 MiB), `prompt_path` (when `-f` used), `prompt_sha256`, `command_line` (real argv, was placeholder until this round), `started_at` / `ended_at` (ISO 8601 UTC), `cli_path`, `cli_build`, `thinking_mode`, `reasoning_effort`, `max_new_tokens`. All round-trip through `RunMetadata.from_dict`.)
- ✅ Scripts write results to timestamped files or require explicit overwrite confirmation. (`run_glm52_moe_trace.sh` suffixes output paths with `-$(date +%Y%m%d-%H%M%S)`; `run_trace_suite_batched.sh` writes per-prompt traces into a timestamped batched dir. The C++ writer refuses to overwrite existing files.)
- ✅ Reports include the exact input trace files used. (`## Reproducibility provenance` section now includes full `command_line` + `sources[]` list with trace filenames. The JSON summary also carries all provenance fields per source.)
- ✅ Project memory files are updated with major findings. (`GLM52_SESSION_MEMORY.md` is the canonical findings record per AGENTS.md; `GLM52_TRACE_PLAN.md` session-status section tracks phase progress; both updated each round.)
- ✅ Trace artifacts are stored under predictable `traces/` and `reports/` paths. (Traces: `traces/<batch_name>/<test_id>-<language>.jsonl`. Reports: `reports/glm52_<study_name>_report.md` + `_summary.json`. Both directories documented in `traces/README.md`.)

### Story 10 — CI maintainer validates quantization/pruning scripts without rewriting models

**As a CI maintainer, I need quantization and layer-pruning dry-runs to behave like idempotent integration tests, so that GitHub Actions can catch script, dependency, llama.cpp CLI, and GGUF plan regressions without producing model artifacts.**

Acceptance criteria:

- ✅ `mixed-precision-quantization/scripts/quant_glm52_mixed.sh --dry-run` validates source shards, imatrix, tensor-type mapping, GGUF metadata scan, and native `llama-quantize --dry-run` with the same planned options, while leaving `/Volumes/Data NVME/GLM-5.2-GGUF/glm52_tensor_types.txt` unchanged. (Verified 2026-06-21: native dry-run emitted model size 355388.74 MiB and quant size 237634.13 MiB; tensor-types mtime unchanged.)
- ✅ `layer-level-structured-pruning/scripts/prune_layers.py --dry-run` eagerly imports `prune_gguf`, validates its hook API, loads the BI plan, scans GGUF shards, and reports dropped/kept tensor counts plus `blk.78` renumbering without creating output directories or shards. (Verified 2026-06-21: 1809 total tensors, 368 dropped, `split.tensors.count` 1809→1441, MTP→blk.62.)
- ✅ Both dry-runs use standard CI semantics: exit `0` only when all checks pass; any non-zero exit means failure or invalid invocation and stderr explains the error. (Verified 2026-06-21: valid dry-runs exit 0; missing prune args use argparse exit 2.)
- ✅ A GitHub Actions workflow runs lightweight syntax checks on GitHub-hosted Ubuntu, then runs the real-model quant/prune dry-runs on a self-hosted macOS runner labeled `glm52` with local GLM-5.2 artifacts. The workflow uses `uv run --with gguf --with numpy python` for Python checks/dry-runs and builds the patched llama.cpp binaries on the runner if `vendor/llama.cpp/build-metal/bin/llama-quantize` is missing. (`.github/workflows/glm52-dry-run.yml`.)

## Definition of done for Phase 1

Phase 1 is done when:

- A traced GLM-5.2 run produces structured MoE routing JSONL.
- The analyzer produces a readable report for at least three prompt categories.
- The current mixed GGUF baseline can run with tracing enabled.
- Tracing disabled mode remains unchanged.
- Trace JSONL writing is asynchronous and does not perform disk I/O from the eval callback.
- Trace output is bounded and does not create unmanageable files.
- Smoke runs record thinking mode and default to disabled/minimal thinking.
- Results are documented in project memory.

## Initial prompt/task suite

Use short and medium prompts first.

A concrete multilingual smoke suite has been created:

```text
prompts/tracing/glm52_trace_smoke_suite.json
prompts/tracing/glm52_trace_smoke_suite.expanded.jsonl
prompts/tracing/README.md
scripts/tracing/expand_smoke_suite.py
```

Smoke-suite counts:

```text
23 base tests
7 languages
161 expanded prompt records
```

Domain distribution:

```text
coding:            4 base tests / 28 expanded prompts
physics:           4 base tests / 28 expanded prompts
math:              3 base tests / 21 expanded prompts
engineering:       3 base tests / 21 expanded prompts
computer_science:  3 base tests / 21 expanded prompts
chemistry:         3 base tests / 21 expanded prompts
cybersecurity:     3 base tests / 21 expanded prompts
```

Language distribution:

```text
en, it, zh, es, fr, de, pt — 23 prompts each
```

Additional future prompt groups:

1. Scientific reading: paste a longer technical abstract or section and ask for key claims.
2. Explanation: ask the model to explain a dense technical paragraph to an engineer.
3. Code-switching prompts: mix English with Italian, Chinese, Spanish, French, German, or Portuguese to observe transition behavior.
4. Long-context retrieval: use the existing 18.7k sentinel retrieval prompt in sampled mode.

Existing useful prompt:

```text
long_coding_task_20k_retrieval_prompt.md
```

## Open questions

- Should tracing be implemented as a new standalone llama.cpp example or as flags in `llama-cli`?
- How much overhead is acceptable for full GLM-5.2 runs?
- Do we need token text in traces, or are token IDs enough for privacy/storage?
- Should DSA indexer tracing be Phase 1b or Phase 2?
- ~~What minimum prompt set is enough to claim task-specific expert specialization?~~ **ANSWERED (2026-06-20, scaled study):** multiple test variants per language×domain cell are required. The 49-prompt one-per-combo grid produced false specialization signals (zh appeared fully disjoint from Latin languages; reversed to 0.11–0.33 overlap at 161 prompts). A single prompt per cell is a fast first look; trust a per-cell claim only after replication across ≥3 prompts in that cell. N≥160 prompts (≈23 per language, 3-4 per cell) gave stable overlap structure.

## Session status (live)

Updated continuously as work progresses. Newest entries at the top.

### 2026-06-20 — Phase 4 / Story 6 bounded activation summaries

Status: **DONE.** All 5 Story 6 ACs closed. Implemented the C++ tracer side
+ Python analyzer side across two commits.

**Python side (commit 6aece97):** Schema got `EVENT_ACTIVATION_SUMMARY =
"activation_summary"` alongside existing `EVENT_MOE_ROUTING`. New
`ActivationSummaryRecord` dataclass: tensor_stem, n_channels, topk,
top_k_channels (list of [channel_idx, magnitude] by |mag|), l2_norm / mean /
std / max_abs per-token stats. Same provenance fields as MoeRoutingRecord
(run_id, model, phase, token_index, layer, task_label, language, script,
prompt_family, test_id) so analyzer can aggregate by task/lang/layer.
`iter_records()` signature widened to `MoeRoutingRecord | ActivationSummaryRecord`
union, dispatches by event field. `Aggregated.activation_summaries` stores
activation records separately from routing. `build_summary()` emits a new
`activation_summary` section: per (task, layer, tensor_stem) row with n_tokens,
mean L2/mean/std/max_abs, top-N channels by frequency-of-appearance.
`render_markdown()` adds `## Bounded activation summaries (Phase 4)` table.
Synth generator got `generate_activation_records()` +
`write_synth_trace(activations=True, activation_stems=..., activation_topk=N)`
so the analyzer pipeline can be tested end-to-end without loading the real
model. Test suite went 78 → 87 (9 new tests covering schema/validation/
round-trip/iter_records dispatch/synth end-to-end).

**C++ side (commit c06d484):** New `TraceConfig` fields: `trace_activations`
(comma-separated stems), `trace_activation_topk` (default 10),
`trace_activation_stride` (default 2). New CLI flags
`--trace-activations <stems>` / `--trace-activation-topk N` /
`--trace-activation-stride N` (pre-scanned by `config_from_trace_flags` so
`common_params_parse` doesn't choke). `trace_cb_eval` dispatches: if
`is_activation_tensor(name, st.activation_stems, matched_stem)` returns true,
compute per-token stats and push `render_activation_record()`; do not fall
through to the routing-event path. Both event types coexist in the same JSONL.

Performance design decisions worth recording:
- **Min-heap top-K**, not `std::partial_sort` per token. 6144 channels ×
  per-token on prefill would make a sort-based approach O(N log N) per token
  = O(N² log N) over prefill. Min-heap of size topk with `std::make_heap`
  upfront + `pop_heap`/`push_heap` on each candidate is O(N log topk) —
  the topk is small (5–50), so log topk ~ ≤6. Net: one heap op per channel,
  two heap ops per replacement. This is the standard top-N-from-stream pattern.
- **Single forward pass** for l2_norm / mean / std / max_abs (sum + sumsq +
  running max_abs in one loop over channels). Variance clamped to ≥0 for
  numerical robustness on all-zero tokens. Net per-token work: N-thats-1
  passes over N channels + an N-log-topk heap walk — same big-O complexity,
  lower constant factor than a 2-pass mean-then-variance approach.
- **Stride** defaults to 2 (emit only for even layers → 1/2 prefill JSONL
  volume), pairs with `--trace-max-tokens` for per-phase token budget. On a
  20k-token prefill through 79 layers with stride 4 → ≤ (79/4) ×
  min(max_tokens, 20k) records — bounded.

Bug found and fixed during implementation:

- **`json_escape_append` uses `s += ",\"run_id\":"; json_escape_append(...);
s += \"\";` — I forgot the opening `\"`.** The result was invalid JSON
  (string values had no opening quote): `{"run_id":act_smoke-en-..."` instead
  of `{"run_id":"act_smoke-en-..."`. Capture: the Python analyzer failed with
  `JSONDecodeError: Expecting value: line 1 column 59` — clean signal that
  JSON was malformed. Fix: copy the exact `\":\"` pattern from
  `render_record()` (the existing moe_topk renderer uses `s += ",\"run_id\":\"";`
  — two quotes, one closing the key, one opening the value). Lesson: when
  mirroring an existing JSONL-render pattern, carefullDIFF of the exact
  quote pattern rather than copying just the close-quote half.

- **String-literal compare warning:** `(st->current_phase == "generation")`
  where `current_phase` is `const char *` triggered
  `-Wstring-compare` (comparison against string literal is unspecified).
  Fix: `std::string(st->current_phase) == "generation"`. Per-token cost
  so it would be better to make `current_phase` a `std::string` field in
  TraceState. Adventure for a future cleanup; the conversion is cheap
  relative to the channel scan.

- **Unused `n_total` variable:** lifted from the copy-paste coverage of the
  MoE case. Removed.

**Verification on real GLM-5.2 mixed GGUF (12-token smoke, stride=4, topk=5):**
```text
records: 2 routing, 6 activation_summary
first activation record top_k_channels:
  [[822, -0.0705869], [4270, 0.0702773], [2864, 0.0581652], ...]
first activation record stats:
  l2_norm=0.833161 mean=0.000259479 std=0.0106261 max_abs=0.0705869
tensor_stem unique: ['l_out']
phases seen: ['generation', 'prefill']
```

Analyzer applied to the real trace produced a real activation-section table:
```markdown
## Bounded activation summaries (Phase 4)
- Activation summary records: **6** across **2** (task, layer, tensor) groups
| task | layer | tensor_stem | topk | n_channels | n_tokens | mean L2 | ... | top channels |
| coding | 0 | l_out | 5 | 6144 | 25 | 0.62 | ... | #4386, #506, #822, #5652 |
| coding | 4 | l_out | 5 | 6144 | 5  | 0.5052 | ... | #4386, #2305, #506, #4801 |
```

Channel #4386 came up top in both layer groups (layer 0 and layer 4) —
first real semantic hint from bounded activation summarization on the real
GLM-5.2 model. Whether #4386 is task-specific (coding) or a general coding-
related channel needs more prompts to disentangle (same sampling-artifact
lesson as the 49→161 monolingual routing study). That’s a Phase 4b question.

Python pipeline tested end-to-end before the C++ side landed: 161 synthetic
traces with `activations=True, n_layers=20, n_prefill=4, n_gen=2, topk=5`
→ 9660 activation_summary records + 966 routing records → analyzer
produced 70 (task, layer, tensor_stem) rows with distinct top-N channel
sets per task/language (chemistry/coding/math/etc. all varied).

**AC audit after this round:** 41 ✅ + 5 new = **46 ✅, 4 ⬜**. The 4
remaining open ACs are ALL in Story 5 (DSA / long-context retrieval
tracing) which is hard-blocked at the llama.cpp forward-graph layer (see
Phase 3 finding earlier in this memory). All Phase 1, 2, 4 ACs are now
closed. Only Phase 3 remains blocked.

### 2026-06-20 — Phase 3 / Story 5 DSA tracing blocker (empirically confirmed)

Status: **HARD-BLOCKED at llama.cpp forward-graph layer**, not at the
REAP37/mlx-lm layer the prior audit cited. Empirical probe.

Prior audit (last 2 rounds) deferred Story 5 with rationale "Phase 3 work; not
yet implemented. Gated on REAP37 IndexShare unblock in mlx-lm OR confirmed DSA
tensor visibility in the GGUF baseline." The OR was wishful — I didn't verify
which side of the disjunction actually applies. Investigation this round:

**1. The GGUF DOES carry the indexer tensors.**
Shard 2+ of GLM-5.2-mixed GGUF has 60 `blk.N.indexer.*` tensors across
layers 0..14 (full set: `blk.0.indexer.attn_k.weight`, `blk.0.indexer.k_norm`,
`blk.0.indexer.proj.weight`, etc.). Confirmed via:
`python3 -c "import gguf; r=gguf.GGUFReader(shard2); [t.name for t in r.tensors if 'index' in t.name.lower()]"`

**2. The arch loads them correctly into the layer struct.**
`src/models/glm-dsa.cpp` lines 104-108 populate `layer.indexer_k_norm`,
`layer.indexer_proj`, `layer.indexer_attn_k`, `layer.indexer_attn_q_b` via
`create_tensor(tn(LLM_TENSOR_INDEXER_*, ...))`. Hparams
`indexer_n_head=32`, `indexer_head_size=128`, `indexer_top_k=2048` all read.

**3. BUT the forward pass NEVER executes the indexer path.**
`llama_model_glm_dsa` (line 152 of glm-dsa.cpp): `using graph =
llama_model_deepseek2::graph;` — i.e., GLM-DSA aliases **deepseek2's**
forward graph, NOT deepseek32's. `grep -cE indexer src/models/deepseek2.cpp`
returns **0**. The DSA indexer code path lives only in
`src/models/deepseek32.cpp` (lines 110-114 tensor registration, 172-176
hparam reads, 224 `indexer_q`, 293 `indexer_weights`, 343 `indexer_top_k`).
And `src/llama-kv-cache.cpp:340` gates the DSA KV-cache Hadamard rotation on
`LLM_ARCH_DEEPSEEK32` specifically — it would NOT fire for `LLM_ARCH_GLM_DSA`.

**4. Empirical confirmation against the real model.**
Instrumented `trace_cb_eval` to log every tensor name the callback sees during
a 1-token forward pass through layers 0..14:

    823 unique tensor names sampled (URL: /tmp/indexer_probe.log)
      45 Qcur, 43 fattn_mla, 30 Kcur, 30 __fattn__, 25 ffn_norm,
      22 ffn_moe_weights_norm, 22 ffn_moe_weights, 22 ffn_moe_weighted,
      22 ffn_moe_probs, 15 Vcur, 15 q_pe, ..., 14 l_out, 14 kqv_out,
      14 ffn_up, 14 ffn_swiglu
    Matching 'index' / 'dsa' / 'ret' / 'sparse': 0

No DSA / indexer / sparse-retrieval tensor in the entire graph. Full unique-list
saved to session memory. Conclusion: the model loads `blk.N.indexer.*` weights
but they sit in memory unused. GLM-DSA runs as **plain MLA attention** in
llama.cpp today (which is presumably why the baselines still produce correct
output — the model has fallback behavior — but it's running suboptimally,
NOT running the sparse retrieval the architecture was designed for).

**5. Implication for tracer-side work.**
The `cb_eval` callback fires for every tensor in the graph, so when llama.cpp
migrates `llama_model_glm_dsa::graph` from `deepseek2::graph` →
`deepseek32::graph` (and extends the `LLM_ARCH_DEEPSEEK32`-gated paths in
`llama-kv-cache.cpp`), then named intermediate tensors like `indexer_topk_N`
will start appearing in the callback — and the tracer can intercept them
exactly the same way it intercepts `ffn_moe_topk-N` today. **NONE of that is
work we can do from the tracer side** without patching the upstream graph.

**6. Same gap across both runtime backends.**
- mlx-lm: `glm_moe_dsa` doesn't recognize `IndexShare` tensors → model fails
to load (covered in REAP37_EXPERIMENTS.md).
- llama.cpp: `glm_dsa` aliases `deepseek2::graph` (no indexer refs) instead of
`deepseek32::graph` (full indexer path) → model loads but indexer weights are
unused. Same root cause (indexer forward path missing), different failure
mode (silent ignore vs fail-to-load).

**Documented the exact upstream change** that would unblock Phase 3 (in the
GLM52_TRACE_PLAN.md Story 5 AC notes):
  1. In `src/models/glm-dsa.cpp` line 152, change `using graph =
     llama_model_deepseek2::graph;` → `using graph =
     llama_model_deepseek32::graph;`.
  2. In `src/llama-kv-cache.cpp:340`, extend the gate to include
     `LLM_ARCH_GLM_DSA` so the DSA KV-cache Hadamard rotation also fires.
  3. Verify tensor mappings (the deepseek32 graph consumes
     `indexer_proj` / `indexer_attn_q_b` / `indexer_attn_k` / `indexer_k_norm`
     — same names glm_dsa already loads, so mapping should be safe).

Note that this upstream patch is a separate engineering effort and **belongs
in the patched llama.cpp branch** (`feature/patch_used_to_create_mixed_quantization_of_glm5.2`),
not in the tracer work. Spawning it now would mix forward-pass behavior changes
with the tracer instrument — bad idea. Better to do it as its own commit
series when there's appetite to validate forward-pass correctness.

**Phase 3 status:** All 4 Story 5 ACs remain ⬜ but are now annotated with the
precise empirically-confirmed blocker. **No tracer-side action can unblock them
without patching llama.cpp upstream.** When the upstream unblock lands, the
tracer will need ~50 LoC to capture `indexer_topk-N` tensors + a new event
type `dsa_retrieval` in the schema + analyzer distance-bucketing code — design
is mapped out in the Story 5 AC notes.

### 2026-06-20 — Story 8 missing-expert + speed metrics ACs (Phase 2b closure)

Status: **DONE.** Implemented Story 8's last 2 open ACs that I had wrongly
deferred in the prior round's audit. The deferral rationale was wrong both
times:

1. "Reports flag missing experts" — I had marked this ⬜ claiming compare
   diff logic was "not yet implemented". **It was already there** in
   `compare_trace_reports.py` ("Missing experts" + "Expert-count changes" +
   "Speed" sections all exist). What was missing was the *upstream feed*: the
   C++ tracer never populated `n_expert_total` anywhere — the field was
   always 0 in records and sidecar.

2. "Reports include speed metrics" — same pattern: the compare report's
   "Speed" section was reading `perf_gen_per_sec` from each source's meta
   sidecar. The C++ tracer never populated `perf_gen_per_sec`. So the speed
   section always showed `None`.

The fix was upstream-only: populate the two fields in `trace-moe.cpp`.

**Story 8 AC 8.3 — missing-expert diff upstream fix (C++):**
There's no public llama API for "how many experts per layer". The hparams
field `n_expert` is private/experimental and lives in `src/llama-hparams.h`.
But the GGUF KV `<arch>.expert_count` is always written by
`llama_model_saver.cpp` (line 220 of llama-model-saver.cpp: `add_kv(
LLM_KV_EXPERT_COUNT, hparams.n_expert)`) and there's exactly one such key
per GGUF file (one arch per file).

Public API path: `#include "gguf.h"` + `gguf_init_from_file(path, params)`
+ iterate `gguf_get_n_kv(ctx)` keys, match suffix `.expert_count`, read
`gguf_get_val_u32`. Cheap (metadata-only, `no_alloc=true`). For GLM-5.2
this returns 256. Populated once at startup in `main()` since it's a global
per-model value (all MoE layers in one GGUF share the same expert_count).

Each routing record's `"n_expert"` field (line ~243 in `render_record`,
already gated on `n_expert_total > 0`) now emits `256` per event. The
sidecar also carries top-level `n_expert_total: 256`.

**Story 8 AC 8.4 — speed metrics upstream fix (C++):**
llama.cpp has `llama_perf_context(ctx)` and `llama_perf_context_reset(ctx)`
as public APIs (include/llama.h line 1542/1544). `llama_perf_context_data`
holds `t_p_eval_ms`, `t_eval_ms`, `n_p_eval`, `n_eval` (and `t_load_ms`,
not needed here). `perf_reset` zeroes prompt/gen perf but preserves
`t_load_us` (model load time).

`run_one_prompt` now calls `llama_perf_context_reset(ctx)` at the start of
each prompt so per-prompt timings are isolated across batched prompts, and
reads `llama_perf_context(ctx)` after the decode loop to compute:
- `perf_prompt_eval_per_sec = n_p_eval * 1000 / t_p_eval_ms`
- `perf_gen_per_sec         = n_eval * 1000 / t_eval_ms`

Sidecar emits both per-sec values plus the raw `perf_*_ms` and `perf_n_*`
counters for downstream debugging. Python `RunMetadata` dataclass extended
with `perf_prompt_eval_ms`/`perf_eval_ms`/`perf_n_prompt_eval`/`perf_n_eval`/
`n_expert_total` as Optional defaults (zero breaking change for old
sidecars).

**Verification:**
- Real 12-token smoke against GLM-5.2 produced sidecar with real values:
  `n_expert_total: 256`, `perf_prompt_eval_per_sec: 6.2278`, `perf_gen_per_sec:
  0.9171`, `perf_n_eval: 1`, `perf_n_prompt_eval: 10`.
- Analyzer provenance block in markdown now renders:
  `Speed: **0.92 gen tok/s**, 6.23 prefill tok/s (1 gen tokens / 10 prompt tokens)`
  and `n_expert_total: **256** (total routed experts per MoE layer)`.
- Compare report Speed section renders real values when populated (verified
  against real-vs-real-copy: `0.92` mean gen/s for both labels).
- Compare report Missing experts section: 0 missing for identical-content
  labels — correct (would flag any expert in label A's union but absent in B's).
- 78 tests pass (was 76; +2 round-trip tests in TestRunMetadataStory8PerfAndExpertCount),
  ruff clean, C++ builds warning-clean.

AC audit after this round: 39 ✅ + 2 new = **41 ✅, 9 ⬜**. The 9
remaining are all hard-gated future-phase work:
- Story 5 (4): DSA long-context retrieval tracing — Phase 3, blocked on
  REAP37 IndexShare unblock.
- Story 6 (5): bounded activation summaries — Phase 4, schema's `event`
  discriminator ready for an `activation_summary` type.

All actionable Phase 1+2 ACs are now closed. No implementation work left in
Phase 2-territory.

### 2026-06-20 — Story 9 reproducibility AC + Story 1-9 audit

Status: **DONE.** Audited all nine user stories' acceptance criteria (was: only
Story 7 had been audited previously). Result: 39 ✅, 11 ⬜. The 11 remaining
are all genuinely deferred Phase 3/4 work or require a real second-model variant:
- Story 5 (4 open ACs): DSA / long-context retrieval tracing — Phase 3, gated
  on REAP37 IndexShare unblock.
- Story 6 (5 open ACs): bounded activation summaries — Phase 4, gated on
  Phase 3 wrap-up. The schema's `event` discriminator is already in place to
  add an `activation_summary` event type.
- Story 8 (2 open ACs): reports flag missing experts + include speed metrics.
  Missing-expert flagging requires `n_expert_total` to be captured in each
  record (currently only `n_expert_used=8` is). Speed metrics require a real
  comparison variant — only candidate (REAP37 MLX compat) is marked INVALID.

Implemented Story 9's primary AC: real provenance fields (was placeholder
strings) in the C++ `.meta.json` sidecar. The C++ tracer now emits:
- `command_line` (real joined argv, was `"llama-trace-moe ..."`)
- `prompt_sha256` (real FIPS-180-4 hash of `params.prompt`, was `"(see run log)"`)
- `model_sha256_prefix` (first 1 MiB, 16 hex chars — cheap provenance)
- `model_total_size_bytes` (249186991232 = 232 GiB across all 9 shards; the
  per-shard `model_size_bytes` of shard 1 looked misleadingly tiny at 9.4 MiB
  because shard 1 only carries the GGUF header)
- `started_at` / `ended_at` (ISO 8601 UTC timestamps)

Added `RunMetadata.from_dict()` classmethod (was inline in `load_meta_sidecar`; 
migrated loader to use the cleaner API) and a `model_total_size_bytes` optional
field. Two new tests: round-trip of all new fields + backward-compatibility
with old sidecars that carry placeholders. 76 tests pass, ruff clean.

Analyzer report now opens with a **Reproducibility provenance** block: full
command line for re-running, model name, file sizes, hashes, and run window —
so any analyst can reproduce a trace from a markdown report alone (Story 2/9).

Verified end-to-end against the real GLM-5.2 mixed GGUF: 8-token repro smoke
run produced a sidecar with real values (prompt_sha256
`7ba5a486...`, model_total_size_bytes 232.1 GiB, run window 2026-06-20T19:02-19:03 UTC).
Synth pipeline (`make_synth_trace.py` → `analyze_moe_trace.py` →
`compare_trace_reports.py`) verified intact: 161 synth traces → 6440 records →
report with new provenance block.

C++ tracer rebuilt warning-clean. Both repos ready to commit.

### 2026-06-20 — Code-switch routing study + Story 7 AC complete

Status: **DONE.** Authored `prompts/tracing/glm52_code_switch_suite.expanded.jsonl`
(16 prompts: 6 lang pairs × 3 domains + 1 triple en+zh+es) and traced it via
the batched wrapper. 51,554 records, 0 dropped, 75 layers, 5.1 min. Marks
Story 7's last open AC (code-switching) ✅ — fully closes Story 7.

Fixed Bug 7: bash `${VAR:-default}` treats empty string as null, so the wrapper
silently fell back to the default 7-language list when `LANGS=""` was set — which
filtered out every code-switch label. Switched LANGS/DOMAINS to `${VAR-default}`
(no colon before dash): unset → default, empty → no filter. Backward-compatible.
(See `GLM52_SESSION_MEMORY.md` "Bug 7" for full RCA.)

Scientific findings documented in `GLM52_SESSION_MEMORY.md` "Code-switch routing
study":
1. Code-switch entropy = midpoint of the two component languages (-0.015 to
   +0.017 bits from midpoint — within noise). Mixing two languages in one prompt
   does NOT raise router uncertainty.
2. Code-switch top experts are a partial union of the two languages' monolingual
   top experts: ~5-7 of the 13-15 expert union overlap, split roughly evenly
   with each component language. New routing distribution, neither copy nor
   union.
3. English remains the lowest-entropy language even within code-switch pairs
   (router more deterministic when English is involved).

16-prompt N is small (2-3 per cell); same 49→161 lesson applies: per-cell claims
need replication. Treat as framework-validation + first-look data.

### 2026-06-20 — RESOLVED: generation-phase MoE tensor readback garbage

Status: **RESOLVED.** Found by running the suite wrapper (now that it works)
and noticing generation-record expert IDs were garbage (`1347970970`, …) with
entropy 0, while prefill was clean.

Root cause was a real Metal readback bug with an important lesson:

- The original `trace_cb_eval` read MoE tensor data with
  `if (ggml_backend_buffer_is_host(t->buffer)) data = t->data; else
  ggml_backend_tensor_get(...)`. On the Metal backend, intermediate MoE tensors
  can live in shared host memory whose GPU compute write has NOT landed by the
  time the eval callback fires. During prefill (large batch) a natural sync
  point masked this; during fast single-token generation decodes the stale-read
  window was hit, producing garbage `expert IDs` and zeroed weights.
- Canonical `common_debug_cb_eval` (common/debug.cpp) has the SAME
  `is_host → t->data` shortcut, so it would exhibit the same staleness on Metal
  generation — the difference is it's a debugging tool tolerant of stale data,
  whereas a tracer must not be.
- A secondary bug I introduced while fixing the first: I assumed both MoE tensors
  are 2D `ne=[n_used, n_tokens]` and switched to stride-aware reads. The
  weights tensor is actually 3D `ne=[1, n_used, n_tokens]` (
  `nb=[4,4,n_used*4,...]`), so reading `ne[0]` as n_used gave 1 and zeroed
  elements 1..7. Reverted to the flat read using the topk's `n_used` for BOTH
  tensors — the copied bytes are byte-contiguous in the `[n_used, n_tokens]`
  frame for GLM-5.2.

Fix (final): ALWAYS `ggml_backend_tensor_get` into a fresh per-call buffer
(never read `t->data` directly, even when `is_host`), then flat-index weights
using the pending topk's `n_used`/`n_tokens`. Pure correctness fix; no schema
or output format change.

Re-verified on the real model (2-prompt en/zh suite): prefill entropy
2.72–3.0 with correct 8 weights `[0.968, 0.291, 0.296, …]` (matches the
original clean sample); **generation entropy 2.5–2.98** (was 0) with distinct
softmax weights `[0.569, 0.819, 0.750, …]`, 0 bad records in both phases,
0 dropped, 32 layers captured (was 4).

### 2026-06-20 — Phase 2b-scaled multilingual study (161 prompts) **CORRECTS the 49-prompt headline**

Status: **DONE.** Re-ran the multilingual study with the FULL expanded suite
(161 prompts, 3-4 test variants per language×domain cell) instead of the
one-per-combo 49-prompt subset. **590,467 routing records, 0 dropped, 75 layers,
41.6 min wall.** Findings in `GLM52_SESSION_MEMORY.md` "Scaled multilingual
study" section. Artifacts: `reports/glm52_multilingual_full_report.md` +
`_summary.json`.

**The 49-prompt headline ("zh shares zero top experts with Latin languages")
was a sampling artifact.** At 161 prompts zh overlaps substantially with Latin
languages (en|zh=0.333, de|zh=0.25, pt|zh=0.25, fr|zh=0.176, es|zh=0.111,
it|zh=0.0). The one-prompt-per-cell grid produced false specialization signals.
This answers the open question "What minimum prompt set is enough to claim
task-specific expert specialization?": **multiple test variants per cell are
required; a single prompt per cell is a first look only and must not be the
basis for a per-cell claim.**

Findings that **held at scale** (robust):
- **Prefill vs generation fully disjoint** (top-10 Jaccard 0.0, down from 0.05):
  prefill #169×14760/#247/#199/#106/#99, generation #39×7592/#171/#23/#221/#71.
  The most reliable routing signal in the study.
- Router entropy ~uniform 2.76–2.82 bits (en lowest, de highest; chemistry
  lowest task, cybersecurity highest). Specialization = WHICH experts, not
  entropy.
- Romance cluster: es|pt=0.333 highest; example tokenization efficiency en=29,
  zh=28.7, de=43.3 mean prompt tokens.
- Task-domain semantic overlap: chemistry|math, coding|computer_science,
  coding|math, computer_science|cybersecurity all = 0.429.

Findings that **weakened at scale** (false positives at N=49):
- Language specialization (zh disjoint, de|en=0.0) — gone at scale.
- Expert specialization per task: engineering/physics dropped 0.5 → 0.2; only
  cybersecurity and physics retain 0.4.

The 49-prompt one-per-combo grid is still useful for fast first looks (~12 min
vs ~42 min for the full 161), but its per-cell claims need replication across
≥3 prompts before being trusted.

### 2026-06-20 — Phase 2b multilingual routing study (49-prompt first results — SUPERSEDED)

Status: **SUPERSEDED** by the 161-prompt scaled study above. The 49-prompt run
(180,457 records, one prompt per language×domain cell) is kept as a
methodological reference point: it demonstrates exactly how a single prompt per
cell can produce false specialization signals. Headline below is the ORIGINAL
49-prompt interpretation; see the scaled entry above for the corrected numbers.

Original 49-prompt headline (now known to be partly sampling noise):
1. Router entropy ~uniform (2.74–2.83 bits, max spread 0.09): specialization is
   in WHICH experts fire, not how peaked the distribution is.
2. Chinese shares zero top experts with any Latin language (all zh-pairs = 0.0)
   — **NOT robust; reversed at 161 prompts**.
3. Task overlap: coding|computer_science=0.43, coding|engineering=0.0.
4. Expert specialization: engineering+physics 0.5 unique, CS 0.1.
5. Prefill vs generation near-disjoint (Jaccard 0.05) — **held and strengthened
   to 0.0 at scale**.

### 2026-06-20 — Phase 1 implemented & verified

Status: **DONE.** Python framework (`src/gguf2mlx/tracing/`), CLI scripts, tests
(74 pass), C++ backend (`examples/trace-moe/`, built warning-clean), real-data
smoke on mixed GLM-5.2 (32 records, 8 experts/event, 0 dropped), synth pipeline
(161 traces → 6440 records → report). Docs updated. See
`GLM52_SESSION_MEMORY.md` → "Phase 1 tracer implementation".

### 2026-06-20 — OPEN: shell wrappers pass example-restricted flags

Status: **RESOLVED.** Found while validating the run scripts that
were referenced in docs/AGENTS.md but never executed (the real-data smoke ran
the `llama-trace-moe` binary directly).

`scripts/tracing/run_glm52_moe_trace.sh` (and by extension
`run_trace_task_suite.sh`, which calls it) built a `trace_args` array that
included `--jinja -cnv -st --chat-template-kwargs`, all `set_examples()`-
restricted to CLI/SERVER/MTMD while the example is `LLAMA_EXAMPLE_COMMON`.

Fix applied (two parts):

1. **C++ pre-scan now strips the 4 chat-template flags** in
   `config_from_trace_flags` and `main()` emits one `LOG_WRN` notice
   (`ignoring chat-template flags … the tracer tokenizes params.prompt
   verbatim`). Chosen over changing the example type to `LLAMA_EXAMPLE_CLI`
   because the tracer never calls `common_chat_init` — accepting the flags
   would have silently done nothing. The truthful behavior (verbatim prompt,
   already documented in `traces/README.md`) is preserved and made explicit.
2. **Wrapper used an invalid `--ngl` flag** (valid forms are `-ngl` /
   `--gpu-layers` / `--n-gpu-layers`; there is no `--ngl`). Fixed to `-ngl`
   to match the baseline scripts.

Re-validated the wrapper end-to-end on the mixed GLM-5.2 model: 10 prompt
tokens → **48 routing records** written (24 prefill + 24 generation), 0
dropped, 0 sampled, distinct prefill-vs-generation router entropy, full
`.meta.json` sidecar + `.run.log` saved. Both run scripts now confirmed working.
