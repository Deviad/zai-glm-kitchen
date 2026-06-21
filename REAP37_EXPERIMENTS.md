# GLM-5.2 REAP37 MLX Experiment Notes

Date: 2026-06-20 CEST
Project folder: `/Volumes/Data NVME/gguf2mlx`

## Goal

Evaluate a REAP-pruned GLM-5.2 variant and compare speed/behavior against the current custom mixed GGUF baseline.

Target model:

```text
https://huggingface.co/pipenetwork/GLM-5.2-REAP37-MLX-4bit
```

REAP project:

```text
https://github.com/CerebrasResearch/reap
```

## Important distinction

The target HF model is already a **REAP37 + 4-bit MLX** artifact. We are not applying REAP locally to our GGUF. We are downloading/testing a prebuilt pruned MLX model in a separate folder.

## REAP37 model facts

From model card / config / index:

```text
Repo:             pipenetwork/GLM-5.2-REAP37-MLX-4bit
Library:          mlx
Model type:       glm_moe_dsa
Architecture:     GlmMoeDsaForCausalLM
Quantization:     4-bit affine, group_size 64
n_routed_experts: 160   # down from 256
num_experts/tok:  8
num_hidden_layers: 78
MTP layers:       1
Advertised size:  265 GB
Index total size: 265,350,012,288 bytes
Parameters:       471,541,876,704 (~472B)
```

Model-card claim:

```text
REAP keeps the 160 most-salient experts per layer out of 256, reducing GLM-5.2 to ~480B params.
```

REAP method summary:

```text
Router-weighted Expert Activation Pruning: experts are scored by mean(router_gate_weight × ||expert_output||) over a calibration set; lowest-saliency experts are dropped and router is sliced to survivors. No retraining.
```

## Local support check

`mlx-lm` 0.31.3 supports GLM-DSA:

```text
mlx_lm.models.glm_moe_dsa
```

So this MLX model should be testable locally.

## Separate local folder

Use this folder for the REAP37 MLX model:

```text
/Volumes/Data NVME/GLM-5.2-REAP37-MLX-4bit
```

Do not mix it with the current GGUF baseline folders.

## Scripts

Created:

```text
scripts/reap37/download_reap37_mlx.sh
scripts/reap37/verify_reap37_mlx.sh
scripts/reap37/run_reap37_merge_sort_baseline.sh
scripts/reap37/run_reap37_longctx_retrieval_baseline.sh
```

### Download

```bash
cd "/Volumes/Data NVME/gguf2mlx"
./scripts/reap37/download_reap37_mlx.sh
```

Default destination:

```text
/Volumes/Data NVME/GLM-5.2-REAP37-MLX-4bit
```

### Verify metadata

```bash
./scripts/reap37/verify_reap37_mlx.sh
```

### Short coding baseline

```bash
./scripts/reap37/run_reap37_merge_sort_baseline.sh
```

Default output:

```text
reap37_mlx_merge_sort_output.txt
```

### Long-context retrieval baseline

```bash
./scripts/reap37/run_reap37_longctx_retrieval_baseline.sh
```

Default prompt:

```text
long_coding_task_20k_retrieval_prompt.md
```

Expected answer:

```text
sentinel: BLUE-FALCON-48217
function: repair_event_stream
recursion_allowed: no
```

Default output:

```text
reap37_mlx_longctx_retrieval_output.txt
```

## Comparison baseline: current custom GGUF

Current known-good GGUF baseline:

```text
/Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/GLM-5.2-mixed-00001-of-00009.gguf
```

Current GGUF baseline results:

```text
Short merge sort:
  Prompt:     ~31.5 tok/s
  Generation: ~20.2 tok/s
  Output: correct iterative merge sort, passed 6 sanity tests

~20k retrieval:
  Prompt:     76.9 tok/s
  Generation: 11.4 tok/s
  Wall time:  278.39s
  Answer: sentinel/function/recursion facts all correct
```

## Caveats

- This is MLX, while the baseline is llama.cpp GGUF, so speed comparison is not perfectly apples-to-apples.
- The REAP37 model is larger on disk than our custom mixed GGUF (advertised 265 GB vs 232 GB), but it has fewer routed experts (160 vs 256), so it may be faster at inference.
- If MLX memory pressure is high, try reducing prompt length or using `--max-kv-size`.

## First REAP37 MLX test results

Downloaded model to:

```text
/Volumes/Data NVME/GLM-5.2-REAP37-MLX-4bit
```

Verified:

```text
247 GB
57 safetensor shards
model_type: glm_moe_dsa
n_routed_experts: 160
quantization: 4-bit affine group_size 64
```

Problem: stock `mlx-lm 0.31.3` and GitHub main do not implement GLM-DSA IndexShare. They instantiate `deepseek_v32` attention with an indexer on every layer and fail to load the REAP37 model because shared IndexShare layers do not have indexer tensors.

Failure:

```text
ValueError: Missing 285 parameters: model.layers.*.self_attn.indexer.*
```

Created compatibility folder:

```text
/Volumes/Data NVME/GLM-5.2-REAP37-MLX-4bit-indexer-compat
```

Compat method:

- hardlink original REAP37 MLX files
- add `model-indexshare-compat.safetensors` (~287 MB)
- duplicate previous Full-layer indexer tensors into shared layers so stock `mlx-lm` can load
- add 627 compatibility tensors

Script:

```text
scripts/reap37/make_reap37_mlx_indexer_compat.sh
```

Short merge-sort baseline on compat:

```text
Prompt:     7.845 tok/s  # tiny prompt, load overhead dominates
Generation: 18.053 tok/s
Peak memory: 265.802 GB
Output: correct non-recursive merge sort; passed 6 sanity tests
```

Long-context retrieval baseline on compat:

```text
Prompt:     109.571 tok/s
Generation: 13.924 tok/s
Peak memory: 281.155 GB
Output: FAILED — generated gibberish, did not retrieve sentinel/function/recursion facts
```

Interpretation:

- REAP37 MLX compat is faster than GGUF on prefill/generation.
- The compatibility workaround is probably semantically wrong for GLM-DSA IndexShare, especially at long context.
- Do not treat REAP37 compat as a valid quality baseline until proper IndexShare support is implemented in `mlx-lm` or a proper GGUF REAP37 exists.

### Additional REAP37 compat sanity test: ~4.9k retrieval

Prompt:

```text
long_coding_task_4k_retrieval_prompt.md
```

Token estimate:

```text
4,855 tokens pre-template
```

Result on REAP37 compat:

```text
Prompt:     163.975 tok/s
Generation: 14.815 tok/s
Peak memory: 270.156 GB
Output: FAILED — gibberish, did not return sentinel/function/recursion facts
```

Conclusion strengthened: the duplicated-indexer compatibility hack allows stock `mlx-lm` to load and answer tiny/simple prompts, but is not semantically valid for retrieval or longer prompts. Proper GLM-DSA IndexShare support is required for meaningful REAP37 MLX quality testing.

## Cross-reference: llama.cpp's glm-dsa has the same gap (2026-06-20)

While investigating whether Phase 3 (Story 5 — DSA tracing) was blocked at the
REAP37/mlx-lm layer or at the llama.cpp layer (summary recorded in
`GLM52_SESSION_MEMORY.md` and `GLM52_TRACE_PLAN.md` Story 5 AC notes), I
patched `trace-moe.cpp` with a temporary debug log of every tensor name the
eval callback sees during a 1-token forward pass against the GLM-5.2 mixed
GGUF baseline through layers 0..14, then grep'd the 823 unique tensor names
for any `index` / `dsa` / `ret` / `sparse` substring.

Result: **ZERO matches.** The forward pass does not run any DSA indexer path.

Root cause mirrored on the mlx-lm side (different failure mode, same gap):

```text
                  mlx-lm                       llama.cpp
                  ------                       ---------
Indexer tensors:  present in REAP37 weights,   loaded (glm-dsa.cpp:104-108
                  but at layers mlx-lm doesn't  creates layer.indexer_k_norm,
                  recognize → fails to load    layer.indexer_proj, etc.)
Forward path:     not implemented               aliased to deepseek2::graph,
                  (no IndexShare support)       which has 0 references to
                                                "indexer anywhere
Failure mode:    crash on missing params       silent — model loads, runs as
                  ValueError: Missing 285       plain MLA; output still passes
                  params: model.layers.*.       the basic baselines (merge
                  self_attn.indexer.*            sort, 20k sentinel retrieval)
                                                but is behaviorally suboptimal
```

The full per-layer tensor list (post-instrumentation) shows only standard MLA
attention tensors (`fattn_mla-N`, `q_nope-N`, `kv_cmpr-N`, `Qcur-N`, etc.) and
standard MoE FFN tensors (`ffn_moe_topk-N`, `ffn_moe_weights-N`, etc.). No
device-level DSA tensors are present.

**Unblock plan for the llama.cpp side** (separate engineering effort from the
tracer work; not yet actioned because it belongs in a forward-pass correctness
patch, not a tracer instrument):

1. In `src/models/glm-dsa.cpp` line 152, change
   `using graph = llama_model_deepseek2::graph;` →
   `using graph = llama_model_deepseek32::graph;` — wires GLM-DSA to the
   graph implementation that actually consumes the indexer tensors (full
   path in `src/models/deepseek32.cpp:110-114, 172-176, 224, 293, 343`).
2. In `src/llama-kv-cache.cpp:340`, extend the DSA Hadamard-rotation gate
   from `LLM_ARCH_DEEPSEEK32` to also fire for `LLM_ARCH_GLM_DSA`, so the
   DSA KV-cache path activates for the GLM-DSA arch.
3. Verify deepseek32's graph consumes the same tensor names glm-dsa loads
   (already confirmed by comparing `load_arch_tensors` in glm-dsa.cpp:106-108
   vs deepseek32.cpp:110-114 — identical tensor registration calls). Mapping
   is safe by construction; the deepseek32 graph was written against these
   tensor names directly.

When the unblock lands, the tracer will start seeing `indexer_*` named
tensors in the `cb_eval` callback and can capture DSA retrievals the same
way it captures `ffn_moe_topk-N` today (~50 LoC tracer + new `dsa_retrieval`
event type). Phase 3 / Story 5 is then tractable.

Note: this finding does NOT change the REAP37-side recommendation — the
compat hack still produces gibberish at long context and is still marked
INVALID for quality comparison. It just documents that the llama.cpp side
has the analogous gap (silently, instead of crashing) so future readers
don't try to validate Phase 3 against the GGUF baseline as-is.
