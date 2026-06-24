# Project Agent Memory: gguf2mlx / GLM-5.2

Always read and respect this file when working in this repository.

## Canonical session record

Detailed GLM-5.2 implementation, quantization, testing, and tracing notes are saved in:

```text
GLM52_SESSION_MEMORY.md
```
If a future task touches GLM-5.2, GLM-DSA, llama.cpp quantization, the mixed GGUF model, or MoE expert-routing tracing, read that file first.

**Findings-tracking contract.** Every round, durable findings must be appended to `GLM52_SESSION_MEMORY.md` (and cross-referenced from `GLM52_TRACE_PLAN.md` when tracing-related), not left only in commit messages or conversation. What counts as a finding worth saving:

- Root-cause analysis of any non-trivial bug (symptom → cause → fix → verified result), especially subtle reuse/concurrency/Metal-backend bugs (e.g. `TraceWriter.open()` must reset `stop`; `trace_cb_eval` must never read `t->data` directly even when `is_host`).
- Real-model experiment results: prompt grids used, record counts, entropy ranges, top experts, layer coverage, wall time, and the headline interpretation (e.g. Phase 2b multilingual study: zh shares zero top experts with any Latin language).
- Decisions and their rationale (e.g. "tracer tokenizes `params.prompt` verbatim, no chat template applied by default").
- Rejected approaches and why.

Format: `### Bug N: <title>` or `### <date> — <topic>` followed by narrative + a **bolded fix** line + a verified-result block, matching the existing entries.

## User stories & acceptance criteria

User stories and acceptance criteria for the planned work already exist in:

```text
PLAN.md            → §7 Acceptance criteria (converter: glm-dsa arch mapping, MLA tensor remap, tokenizer/chat-template, regression)
GLM52_TRACE_PLAN.md → "User stories and acceptance criteria" section (tracer: which experts selected per token/layer, cross-task/cross-language comparison)
```
When starting any new body of work in this repo, follow this tracking contract:

1. **Read** existing stories in `PLAN.md` and `GLM52_TRACE_PLAN.md` first; check whether the new task maps to an existing story before inventing a new one.
2. **Add a user story** ("As a …, I need …, so that …") under the appropriate plan file's acceptance-criteria section before implementing if the work is not already covered.
3. **Add acceptance criteria** as a bulleted list of checkable outcomes (exact paths, function names, expected numerical results, test names) so "done" is objective, not subjective.
4. **Reference the story** in the implementation commit message so progress is traceable story → commit → finding.
5. **Mark the story done** (or note partial completion with remaining criteria) in the plan file when the work lands.
6. **Append a finding** to `GLM52_SESSION_MEMORY.md` per the contract above if the implementation surfaced a bug, insight, or experiment result.

Do not implement a sizable feature without a story + acceptance criteria recorded in one of those two plan files. One-off fixes and typos can skip this.

## Known-good custom GLM-5.2 model

Use this mixed-precision GGUF as the current baseline model:

```text
$MODEL_DIR/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/GLM-5.2-mixed-00001-of-00009.gguf
    (env: MODEL_DIR; see LOCAL_SETUP.md)
```

Quantization policy:

```text
normal routed expert MLPs: IQ2_S
blk.78 MTP routed experts: IQ4_NL exception
all non-experts:           IQ4_NL/F32/Q6_K high precision
```

Verified size:

```text
232 GB / 2.64 BPW
```

Verified tensor mapping:

```text
225 normal routed expert tensors: IQ2_S
3 blk.78 MTP expert tensors:      IQ4_NL
non-expert tensors:               no IQ2
```

## Patched llama.cpp

Tracked as a git submodule at `vendor/llama.cpp` on branch
`feature/patch_used_to_create_mixed_quantization_of_glm5.2` of
`Deviad/llama.cpp`. Fork descends from `ggml-org/llama.cpp`.

`build_llamacpp.sh` builds directly into the submodule by default (it detects
the submodule case via `.git` being a file → skips git fetch/checkout/reset
so the pinned feature-branch state is preserved). `build-metal/` is covered
by llama.cpp's own `.gitignore` (`/build*`) so the submodule status stays
clean after a build.

Use the patched build, not Homebrew `llama.cpp` v9200:

```text
$ROOT/vendor/llama.cpp/build-metal/bin/llama-cli
$ROOT/vendor/llama.cpp/build-metal/bin/llama-quantize
$ROOT/vendor/llama.cpp/build-metal/bin/llama-trace-moe
```
    (`$ROOT` = kitchen checkout root, resolved by each script. Override via
    `LLAMA_SRC` for an alternate build location; `CLI`/`TRACE_BIN` to point
    at a specific binary. See LOCAL_SETUP.md.)

Local patch applied in:

```text
$ROOT/vendor/llama.cpp/src/llama-quant.cpp
```

Patch reason: llama.cpp issue #24379, MTP quantization bug. The quantizer must use `n_layer_all`, not `n_layer()`, for FFN/MoE tensor counters so `blk.78` is valid.

Patch:

```diff
- qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)qs.model.hparams.n_layer();
+ qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)qs.model.hparams.n_layer_all;
```

## Baseline experiments

The following two experiments are the current baselines for further tests.

### Baseline 1: short coding task

Script:

```text
scripts/baselines/glm52_merge_sort_baseline.sh
```

Purpose:

- Load the mixed GLM-5.2 GGUF.
- Ask: `Write down a merge sort algo non recursive in Python`.
- Save output.
- Extract/sanity-test the generated implementation manually if needed.

Previously observed:

```text
Prompt:     ~31.5 tok/s
Generation: ~20.2 tok/s
Output: correct iterative merge sort, passed 6 Python sanity cases
```

### Baseline 2: ~20k-token long-context retrieval

Script:

```text
scripts/baselines/glm52_longctx_retrieval_baseline.sh
```

Prompt file:

```text
long_coding_task_20k_retrieval_prompt.md
```

Exact pre-template token count:

```text
18,745 tokens
```

Expected answer:

```text
sentinel: BLUE-FALCON-48217
function: repair_event_stream
recursion_allowed: no
```

Previously observed:

```text
Prompt:     76.9 tok/s
Generation: 11.4 tok/s
Wall time:  278.39s
Exit:       0
```

## Generation-mode lessons

- Prefer chat mode for GLM-5.2:

```text
--jinja -cnv -st
```

- Raw `-no-cnv` mode with the long coding prompt produced a huge loop of repeated `>` prompts and should not be used as a quality signal.
- The GLM chat template may still emit visible `[Start thinking] ... [End thinking]` even when passing:

```text
--chat-template-kwargs '{"enable_thinking":false,"reasoning_effort":null}'
```

This appears to be a chat-template / llama.cpp behavior, not clear evidence of quantization failure.

## Repro commands

Quantization reproduction scripts live outside this repo in:

```text
zai-glm-kitchen/mixed-precision-quantization/scripts/build_llamacpp.sh
zai-glm-kitchen/mixed-precision-quantization/scripts/quant_glm52_mixed.sh
zai-glm-kitchen/mixed-precision-quantization/scripts/verify_glm52_mixed.py
```

Do not use the older Homebrew `llama-quantize` for GLM-5.2 work.

## REAP37 MLX experiment track

If a task mentions REAP, REAP37, pipenetwork/GLM-5.2-REAP37-MLX-4bit, or CerebrasResearch/reap, read:

```text
REAP37_EXPERIMENTS.md
```

The REAP37 model is a separate **prebuilt MLX** artifact, not the current GGUF baseline:

```text
$REAP37_MODEL_DIR (env: REAP37_MODEL_DIR; canonical default /Volumes/Data NVME/GLM-5.2-REAP37-MLX-4bit)
```

Scripts:

```text
scripts/reap37/download_reap37_mlx.sh
scripts/reap37/verify_reap37_mlx.sh
scripts/reap37/run_reap37_merge_sort_baseline.sh
scripts/reap37/run_reap37_longctx_retrieval_baseline.sh
```

Use these scripts for REAP37 tests so results are reproducible and do not overwrite the GGUF baseline outputs.

### REAP37 status warning

Current REAP37 MLX status:

- Raw downloaded model fails to load in stock `mlx-lm` because `mlx-lm` lacks GLM-DSA IndexShare support.
- `GLM-5.2-REAP37-MLX-4bit-indexer-compat` duplicates missing indexer tensors so stock `mlx-lm` loads, but quality is invalid beyond tiny prompts.
- Short merge-sort works, but both ~4.9k and ~18.7k retrieval tests produce gibberish.
- Do not use the compat folder as a trusted quality baseline. Treat it as a speed/loading experiment only.
- Proper next step is implementing IndexShare in `mlx_lm.models.glm_moe_dsa` or using a proper REAP GGUF if one becomes available.

## DSA / IndexShare forward-path research library

If a task mentions implementing the DSA lightning indexer forward pass, the F/S
(full/shared) IndexShare layer pattern, CSA, or otherwise making GLM-5.2
actually run DSA correctly — note that as of 2026-06-24 the `glm-dsa.cpp`
submodule DOES run its own DSA graph (lightning indexer + `ggml_top_k` on every
layer; AC1/AC2/AC4-AC6 landed), and the F/S IndexShare pattern is CONFIRMED
real upstream (`config.json` carries `indexer_types[]` = 21 full / 57 shared).
The remaining gap is the F/S gating itself (AC3), deferred by design per
REMEDIATION_PLAN.md §P0 (kernel-bound regression + baseline-preservation).
Stock `mlx_lm.models.glm_moe_dsa` still subclasses `deepseek_v32.Model` with no
IndexShare forward path. For the mathematical ground truth read the PDFs in:

```text
    docs/research/papers/   (arXiv PDFs, fetched 2026-06-21, ~10 MB total)
    docs/research/README.md (per-paper abstract quotes + "what to extract")
```

The six papers in that folder are the mathematical ground truth for the gap
above (DeepSeek-V3.2 DSA origin, the IndexCache/IndexShare F/S pattern paper,
GLM-5 tech report, StreamIndex V4 CSA, FlashMemory V4, MISA third-party DSA
repro). They are also cross-referenced in `PLAN.md` §10 and are the basis for
AC7 in PLAN.md §7.M (the honest IndexShare load-time caveat of the mixed-
precision MLX export). `GLM52_TRACE_PLAN.md`'s interpretability work and this
forward-path implementation share the same blockers — both depend on the same
forward-path work, so consult this library first.

## GLM-5.2 tracing / interpretability plan

If a task mentions tracing, debugging activations, expert routing, neurons, interpretability, scientific-document understanding, or coding-vs-science activation comparison, read:

```text
GLM52_TRACE_PLAN.md
```

Initial tracing scope should be MoE expert routing first, not full activation dumps. Preserve the known-good GGUF baseline and write trace artifacts under predictable `common/traces/` and `common/reports/` paths.

**Phase 1 (MoE expert routing tracer) is IMPLEMENTED.** See `GLM52_TRACE_PLAN.md` (top section) and `GLM52_SESSION_MEMORY.md` ("Phase 1 tracer implementation"). The framework lives in `glm52_kitchen/tracing/` (schema, async writer, analyzer, comparator, synth), with C++ backend source in `vendor/llama.cpp/examples/trace-moe/` and binary at `$ROOT/vendor/llama.cpp/build-metal/bin/llama-trace-moe`. Run scripts: `common/scripts/run_glm52_moe_trace.sh` (single prompt), `common/scripts/run_trace_task_suite.sh` (per-prompt-reload wrapper), and `common/scripts/run_trace_suite_batched.sh` (load model once, trace N prompts). Analyze: `common/scripts/analyze_moe_trace.py` / `compare_trace_reports.py`. **Phase 2b (multilingual routing study, 7 langs × 7 domains = 49 prompts) is DONE** — see `common/reports/glm52_multilingual_routing_report.md` and the Phase 2b section in `GLM52_TRACE_PLAN.md`. Phases 2b-scaled, 3/4/5 remain planned.

### GLM-5.2 trace smoke suite

Trace prompt suite lives in:

```text
common/prompts/glm52_trace_smoke_suite.json
common/prompts/glm52_trace_smoke_suite.expanded.jsonl
common/prompts/README.md
common/scripts/expand_smoke_suite.py
```

It contains 23 base tests translated across English, Italian, Chinese, Spanish, French, German, and Portuguese, for 161 expanded prompt records. Domains: coding, physics, math, engineering, computer science, chemistry, cybersecurity.
