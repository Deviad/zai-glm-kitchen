# Phase 3 / Story 5 DSA forward-path patch — empirical trial artifacts

Empirical evidence from the 2026-06-20 attempt to unlock DSA (DeepSeek Sparse Attention)
lightning-indexer tracing on GLM-5.2. The 3-line patch builds clean and activates
the indexer forward path, but produces **garbled gibberish for long-context retrieval**.

Full RCA + diagnosis + decision in
[`../GLM52_SESSION_MEMORY.md`](../GLM52_SESSION_MEMORY.md) → section
"Phase 3 / Story 5 DSA forward-path patch — EMPIRICALLY REJECTED (2026-06-20)".

AC status updates in [`../GLM52_TRACE_PLAN.md`](../GLM52_TRACE_PLAN.md)
→ Story 5 acceptance criteria.

## Schedule

```
1. pre_patch_merge_sort.txt    — pre-patch baseline, ctx=4096, exit 0, correct output
2. pre_patch_longctx.txt       — pre-patch baseline, ctx=32768/18745 prompt, exit 0,
                                  sentinel BLUE-FALCON-48217 recovered
3. (apply 3-line patch, rebuild)
4. post_patch_merge_sort.txt   — patch active, ctx=4096, exit 0, STILL correct
                                  (slowdown: prompt 34.2→28.8 t/s, gen 20.4→8.3 t/s
                                   -59%, expected cost of running the indexer per
                                   decoder layer)
5. post_patch_longctx.txt      — patch active, ctx=32768/18745 prompt, SIGABRT
                                  garbled output from generation token #1
                                  BLUE-FALCON never emitted
                                  chat-template parser assert on malformed output
6. (revert patch, rebuild)
7. post_revert_longctx.txt     — post-revert baseline, ctx=32768/18745 prompt,
                                  exit 0, sentinel recovered, perf matches
                                  pre-patch (prompt 77.1 t/s, gen 11.4 t/s)
```

## The patch that was tried (then reverted)

```diff
--- a/src/models/models.h                  # glm_dsa::graph alias
+++ b/src/models/models.h
-    using graph = llama_model_deepseek2::graph;
+    using graph = llama_model_deepseek32::graph;

--- a/src/llama-model.cpp                  # KV cache dispatcher
+++ b/src/llama-model.cpp
         case LLM_ARCH_DEEPSEEK32:
+        case LLM_ARCH_GLM_DSA:
             res = new llama_kv_cache_dsa(...);

--- a/src/llama-kv-cache.cpp               # Hadamard rotation gate
+++ b/src/llama-kv-cache.cpp
-        if (model.arch == LLM_ARCH_DEEPSEEK32 && ...) {
+        if ((model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_GLM_DSA) && ...) {
```

## Diagnosis (summary)

- Build warning-clean. Patch activates the DSA lightning-indexer forward path —
  the -59% gen t/s on merge-sort proves it (extra mul_mat + Hadamard + ggml_top_k
  per decoder layer).
- At small context (31 prompt tokens), `n_top_k = min(score->ne[0], 2048) = 31`
  → indexer selects all positions → behaves same as plain MLA → output correct.
- At large context (18,745 prompt tokens), `n_top_k = 2048` selects only 2048 of
  18,745 KV positions per query head — but GLM-5.2's `indexer_*` weights appear
  trained with a different (or no) DSA math than DeepSeek-V3.2. Wrong positions
  selected → MLA reads garbage → first generated token is garbage, never recovers.

Either GLM-5.2 has vestigial indexer weights (not used at training time) or uses
a DSA variant incompatible with the deepseek32 graph constructor. Either way,
**tracing DSA on the current GLM-5.2 baseline traces a code path that does not
exist during normal inference.**

## Recommendation

Re-scope Story 5 from "trace DSA indexer top-K selection" to "trace MLA's actual
full-attention patterns at long context using the existing `activation_summary`
event type on `Qcur` / `Kcur` / `q_nope_absorbed` / `kv_cmpr` tensors" — the
mechanism that really does retrieval for this model. Tractable with the current
tracer; would unblock Story 5 once implemented.
