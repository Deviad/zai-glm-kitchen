# PLAN: Add GLM-5.2 (`glm-dsa` / `GlmMoeDsaForCausalLM`) conversion support

**Status:** Ready to implement (pending 3 scope decisions below)
**Mode:** Pure translator — GGUF → HF-named safetensors + config + tokenizer. Does not implement forward passes.

> Originally part of the [`gguf2mlx`](https://github.com/barrontang/gguf2mlx)
> converter repo. Migrated here as part of `zai-glm-kitchen/` so the kitchen
> is self-contained + github-extractable. The implementation of the
> converter itself (`gguf2mlx.convert()`, `detect_architecture()`,
> `extract_and_convert_weights()`, etc.) lives at
> https://github.com/barrontang/gguf2mlx under `src/gguf2mlx/gguf2mlx.py`; tracked
> as a git submodule at `vendor/gguf2mlx` on branch
> `feature/update_for_glm5.2_cooking` of `Deviad/gguf2mlx`.
> This plan tracks the GLM-5.2-specific architecture-mapping work done in
> the converter; for reproducibility of trace/prune/quant experiments
> alongside it, see the matching folders in the kitchen:
> - `mixed-precision-quantization/` for IQ2_S/IQ4_NL policy + scripts.
> - `layer-level-structured-pruning/` for ShortGPT layer-drop + BI analyzer.
> - `common/` for the tracing framework + baselines + reports + prompts.

---

## 1. Goal & scope

Add GLM-5.2 conversion to `gguf2mlx`. GLM-5.2 (`model_type: glm_moe_dsa`, class `GlmMoeDsaForCausalLM`) is a novel architecture combining:

- **MLA** (Multi-head Latent Attention, DeepSeek-V2/V3 family) — low-rank KV cache.
- **DSA** (DeepSeek Sparse Attention) — a lightning indexer that scores compressed KV and selects top-k tokens per query.
- **MoE** — 3 dense + 75 sparse layers, 256 routed + 1 shared expert, top-8, sigmoid + noaux_tc routing.
- **MTP / NextN** — 1 extra block at index `num_hidden_layers` for speculative decoding.
- **IndexShare / IndexCache** — 1-in-4 sparse layers owns the indexer ("Full"); the other 3 ("Shared") reuse the preceding Full layer's top-k indices and carry **no indexer weights**.
- **KVShare** — MTP reuses backbone KV selection (inference-time; no extra converter work).

This repo produces correct *naming/config/tokenizer* output. End-to-end runnability is gated upstream (no `glm-dsa` GGUF producer in llama.cpp master yet; no `mlx-lm` model class yet) — see §6.

---

## 2. Ground truth (verified against published sources)

### 2.1 HF tensor names — from `zai-org/GLM-5.2/model.safetensors.index.json` (282 shards, 1.5 TB)

| Component | HF tensor name pattern |
|---|---|
| MLA | `self_attn.{q_a_proj, q_a_layernorm, q_b_proj, kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj}` (all `.weight`) |
| DSA indexer (F layers only) | `self_attn.indexer.{wq_b, wk, weights_proj}.weight`, `self_attn.indexer.k_norm.{weight,bias}` |
| Dense MLP (layers 0–2) | `mlp.{gate_proj, up_proj, down_proj}.weight` |
| Sparse MoE MLP (layers 3–77) | `mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight`, `mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight`, `mlp.gate.weight`, `mlp.gate.e_score_correction_bias` |
| NextN/MTP (layer 78) | same as sparse layer **+** `shared_head.norm.weight` |
| Root | `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight` |

**Note:** GLM-5.2 HF has a single combined `kv_b_proj` (no split `k_b`/`v_b`). No `k_nope_head_norm`/`k_rope_head_norm`.

### 2.2 GGUF `glm-dsa` tensor enum — from `llama.cpp/gguf-py/gguf/constants.py`

Schema + tensor enum + metadata keys **all exist** in constants.py. No converter class registered yet.

| GGUF tensor name (`blk.{i}.*`) | HF target |
|---|---|
| `attn_q_a` | `self_attn.q_a_proj` |
| `attn_q_a_norm` | `self_attn.q_a_layernorm` |
| `attn_q_b` | `self_attn.q_b_proj` |
| `attn_kv_a_mqa` | `self_attn.kv_a_proj_with_mqa` |
| `attn_kv_a_norm` | `self_attn.kv_a_layernorm` |
| `attn_kv_b` | `self_attn.kv_b_proj` (combined — clean rename) |
| `attn_out` | `self_attn.o_proj` |
| `indexer.attn_q_b` | `self_attn.indexer.wq_b` |
| `indexer.attn_k` | `self_attn.indexer.wk` |
| `indexer.k_norm` | `self_attn.indexer.k_norm` (both `.weight` + `.bias`) |
| `indexer.proj` | `self_attn.indexer.weights_proj` |
| `ffn_gate` / `ffn_up` / `ffn_down` | `mlp.{gate,up,down}_proj` (dense layers) |
| `ffn_gate_shexp` / `ffn_up_shexp` / `ffn_down_shexp` | `mlp.shared_experts.{gate,up,down}_proj` |
| `ffn_gate_inp` | `mlp.gate` |
| `exp_probs_b` | `mlp.gate.e_score_correction_bias` |
| `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` | `mlp.experts.{e}.{gate,up,down}_proj` (per-expert) **or** `mlp.switch_mlp.*` (stacked) — see Issue 2.3 |
| `nextn_shared_head_norm` (block 78) | `model.layers.78.shared_head.norm.weight` |
| `attn_norm` / `ffn_norm` | `input_layernorm` / `post_attention_layernorm` |

### 2.3 Config values — from `zai-org/GLM-5.2/config.json`

MLA: `q_lora_rank=2048`, `kv_lora_rank=512`, `qk_nope_head_dim=192`, `qk_rope_head_dim=64`, `qk_head_dim=256`, `head_dim=192`, `rope_interleave=true`, `rope_theta=8000000`.
MoE: `n_routed_experts→num_experts=256`, `n_shared_experts=1`, `moe_intermediate_size=2048`, `num_experts_per_tok=8`, `first_k_dense_replace=3`, `topk_method="noaux_tc"`, `scoring_func="sigmoid"`, `routed_scaling_factor=2.5`, `norm_topk_prob=true`, `n_group=1`, `topk_group=1`.
DSA: `index_head_dim=128`, `index_n_heads=32`, `index_topk=2048`, `indexer_types[]` (per-layer full/shared), `index_share_for_mtp_iteration`, `mlp_layer_types[]`.
MTP: `num_nextn_predict_layers=1`.
Structural: `num_hidden_layers=78`, `hidden_size=6144`, `num_attention_heads=64`, `tie_word_embeddings=false`, `vocab_size≈151k`.

### 2.4 IndexShare (paper arXiv:2603.12201 = "IndexCache")

Per-layer role string `c∈{F,S}`: F layers own an indexer; S layers have **no indexer tensors**. GLM-5.2 uses 1/4 retention → ~1 in 4 sparse layers is F. **Confirmed in weights**: layer 1 has `indexer.*`, layer 27 has none. The converter must not synthesize indexer tensors on S layers — they simply don't exist in the source.

### 2.5 Tokenizer

BPE (`tokenizer.json`), multi-EOS `[154820, 154827, 154829]`, `pad=151329`, specials `[gMASK]`, `<|user|>`, `<|assistant|>`, `<|observation|>`, `<|system|>`. Chat template ships as `chat_template.jinja`.

---

## 3. Current code state (verified)

- `src/gguf2mlx/gguf2mlx.py`, ~1065 lines.
- `ARCH_MAP` at line 124 — has `chatglm` (line 148), `deepseek2`→`deepseek_v2`, `deepseek3`→`deepseek_v3`. **No `glm-dsa`.**
- `detect_architecture` at line 173.
- `build_config` at line 188 — has a MoE branch (lines 322–350) for qwen2moe/deepseek/dbrx/grok. **No MLA/DSA fields, no glm-dsa branch.**
- `_map_llama_tensor_name` at line 363 — full llama + basic MoE + switch_mlp stacked expert path.
- `_map_tensor_name` at line 447 — **passes `arch` but ignores it**, always calls `_map_llama_tensor_name`. **MLA tensors fall through unmapped.**
- `extract_tokenizer` at line 458 — generic BPE path.
- `convert` at line 902.

**Key insight:** the MLA gap is **pre-existing** and also breaks deepseek2/3 today. Fixing it for glm-dsa retroactively fixes deepseek for free.

---

## 4. Decomposition — issues, root causes, solutions, risks

### Group 1 — Upstream blockers (external)

#### Issue 1.1 — No llama.cpp GGUF producer for `glm-dsa`
- **Problem:** `glm-dsa` enum/schema exist in constants.py but `conversion/glm.py` registers no converter. No canonical GGUF exists.
- **Root cause:** Upstream PR not merged.
- **Solution:** Out of scope. Write converter against the schema; test with synthetic GGUF (§5) + any community fork GGUF when available. No code here depends on the producer existing.
- **Risk:** Low for this repo.

#### Issue 1.2 — No runnable mlx-lm/transformers target
- **Problem:** `mlx_lm/models/` has no glm family; `GlmMoeDsaForCausalLM` in transformers main only.
- **Root cause:** Novel architecture, kernels maturing upstream.
- **Solution:** Acceptance = output matches HF reference layout byte-for-byte at naming/config level, **not** "model loads and runs." Document as experimental.
- **Risk:** Managed by scope discipline.

### Group 2 — Conversion core (this repo)

#### Issue 2.1 — `_map_tensor_name` ignores `arch` (biggest gap)
- **Problem:** Line 447 passes `arch` through, always calls `_map_llama_tensor_name`. MLA/indexer/NextN/shared-expert tensors leak through as `blk.N.attn_q_a.weight`. Breaks deepseek2/3 **today**.
- **Root cause:** Never arch-dispatched.
- **Solution:** Make `_map_tensor_name` a dispatcher:
  ```python
  def _map_tensor_name(gguf_name, arch):
      if arch in ("glm-dsa", "deepseek2", "deepseek3"):
          mapped = _map_mla_tensor_name(gguf_name)
          if mapped:
              return mapped
      return _map_llama_tensor_name(gguf_name)
  ```
  Add `_map_mla_tensor_name` with the MLA table from §2.2.
- **Risk:** Verify no existing deepseek2 path relies on the broken passthrough. Add regression test.

#### Issue 2.2 — MLA `kv_b`: combined vs split is a tensor transform, not a rename
- **Problem:** GLM-DSA emits combined `attn_kv_b` → clean rename. DeepSeek-V3 emits split `attn_k_b` + `attn_v_b`; HF wants single combined `kv_b_proj`. Mapper is name-only; concat must happen in the convert loop.
- **Root cause:** GGUF splits K/V; HF concatenates along output axis.
- **Solution:** In convert loop (line ~902), add per-arch post-step for deepseek2/3: co-locate `attn_k_b`/`attn_v_b`, emit `kv_b_proj = cat([k_b, v_b], dim=0)`. GLM-DSA needs no such step.
- **Risk:** Dimension bookkeeping. Verify `kv_b_proj` row count = `qk_nope_head_dim*n_heads + kv_lora_rank`.
- **DECISION NEEDED:** glm-dsa-only (trivial) or also fix deepseek2/3 (recommended, shared MLA family)?

#### Issue 2.3 — MoE expert layout: `switch_mlp` (stacked) vs `experts.{e}` (per-expert) target mismatch
- **Problem:** Current code (lines 415–422) maps stacked experts → `mlp.switch_mlp.*` (mlx-lm stacked). GLM-5.2 HF reference uses per-expert `mlp.experts.{e}.*`. Mismatch.
- **Root cause:** Two source formats (stacked/per-expert) × two targets.
- **Solution:** For glm-dsa, detect source format and target per-expert:
  - Source per-expert `blk.N.ffn_gate_exps.{e}.weight` → `model.layers.N.mlp.experts.{e}.gate_proj.weight`.
  - Source stacked `blk.N.ffn_gate_exps.weight` [n_exp, …] → split in convert loop into N per-expert tensors (needs `glm-dsa.expert_count`).
  - Keep `switch_mlp` path for qwen2moe/qwen3moe (no regression).
- **Risk:** Highest-complexity change. Confirm source form against first real/fork GGUF. Write per-expert→per-expert rename first; add stacked-split as fallback.
- **DECISION NEEDED:** per-expert `experts.{e}` (matches HF reference — **recommended**) or keep `switch_mlp` stacked?

#### Issue 2.4 — `e_score_correction_bias` mapping
- **Problem:** GLM-5.2 has `mlp.gate.e_score_correction_bias`; no rule exists.
- **Root cause:** Missing rule.
- **Solution:** Map `blk.N.exp_probs_b` (`FFN_EXP_PROBS_B`) → `model.layers.N.mlp.gate.e_score_correction_bias`. Confirm `FFN_EXP_PROBS_B` is in the `GLM_DSA` enum block.
- **Risk:** Low.

#### Issue 2.5 — Shared-expert (`shared_experts`) mapping
- **Problem:** GLM-5.2 has `mlp.shared_experts.{gate,up,down}_proj`; no rule exists.
- **Root cause:** Missing rule.
- **Solution:** Map `blk.N.ffn_{gate,up,down}_shexp` → `model.layers.N.mlp.shared_experts.{gate,up,down}_proj.weight`.
- **Risk:** Low.

#### Issue 2.6 — DSA indexer tensors
- **Problem:** 5 tensors `self_attn.indexer.{wq_b, wk, k_norm.weight, k_norm.bias, weights_proj}` unmapped.
- **Root cause:** New submodule.
- **Solution:** Map: `indexer.attn_q_b`→`self_attn.indexer.wq_b`, `indexer.attn_k`→`self_attn.indexer.wk`, `indexer.k_norm`→`self_attn.indexer.k_norm` (both `.weight`+`.bias`), `indexer.proj`→`self_attn.indexer.weights_proj`. Five rules.
- **Risk:** Low.

#### Issue 2.7 — IndexShare F/S layer handling (GLM-5.2-specific novelty)
- **Problem:** Only F layers have indexer tensors. Naive converter might synthesize or error on missing tensors.
- **Root cause:** Cross-layer weight sharing (IndexCache F/S pattern).
- **Solution:** **Do nothing special — absence is correct.** Convert loop iterates source tensors; S layers have no `indexer.*` source, so none emitted. Read per-layer `indexer_types[]` and `index_share_for_mtp_iteration` verbatim into config. Add **invariant assertion + log**: after mapping, for each layer, presence of `self_attn.indexer.*` must match `indexer_types[i] == "full"`. Mismatch = data error, not silent corruption.
- **Risk:** Low if source GGUF is sparse. Assertion is the safety net.

#### Issue 2.8 — NextN/MTP layer + `shared_head.norm`
- **Problem:** Layer 78 (= `num_hidden_layers`) is MTP block with extra `shared_head.norm.weight`.
- **Root cause:** Extra block beyond transformer layers.
- **Solution:** GGUF block count for glm-dsa = `num_hidden_layers + num_nextn_predict_layers` (per llama.cpp `Glm4MoeModel`). Map `blk.{78}.nextn_shared_head_norm` → `model.layers.78.shared_head.norm.weight`. Rest of block 78 maps like normal sparse layer. Config `num_hidden_layers` must be the **transformer** layer count (78), not total block count (79).
- **Risk:** Medium. **Verify** whether `block_count` metadata reports 79 or 78; expect 79 (Glm4Moe pattern). Config emits `num_hidden_layers=78`, `num_nextn_predict_layers=1`.

#### Issue 2.9 — `build_config` MLA + MoE + DSA fields
- **Problem:** `build_config` (line 188) MoE branch exists but no MLA/DSA/glm-dsa fields.
- **Root cause:** Never extended for this arch.
- **Solution:** Add `glm-dsa` branch emitting (from `glm-dsa.*` metadata): MLA, MoE, DSA, MTP, structural fields per §2.3.
- **Risk:** Low. Add unit test diff'ing output against `zai-org/GLM-5.2/config.json`.

#### Issue 2.10 — Tokenizer + chat template
- **Problem:** GLM BPE multi-EOS, `[gMASK]` specials, jinja template.
- **Root cause:** Generic tokenizer path doesn't write GLM specifics.
- **Solution:** `extract_tokenizer` (line 458) already handles BPE. For GLM: emit `eos_token_id` (primary in config, full array in `generation_config.json`/`tokenizer_config`), write `chat_template.jinja` (from GGUF if present, else canonical GLM template), copy `tokenizer.json` faithfully.
- **Risk:** Medium. Verify multi-EOS representation against HF repo's `generation_config.json`.

### Group 3 — Validation

#### Issue 3.1 — No canonical GGUF to test against
- **Solution:** Build synthetic GGUF fixture with `gguf` lib — 2-layer, 2-expert, F/S-pattern toy exercising every tensor type (dense MLP, sparse MoE, shared expert, MLA, indexer on F, none on S, NextN block). Assert exact HF-name output set + config. Makes converter testable **today**, zero upstream dependency.

#### Issue 3.2 — Regression for existing archs
- **Solution:** Issue 2.1 refactor touches hot path. Add deepseek2 fixture test + re-run smoke tests for qwen2moe/qwen3moe/llama.

---

## 5. Synthetic test fixture design

A self-contained `gguf`-built fixture (tiny dims) covering:
- 1 dense MLP layer (layer 0): `mlp.{gate,up,down}_proj` + MLA + indexer (F).
- 1 sparse MoE layer (layer 1): `mlp.experts.{0,1}.*` + `shared_experts.*` + `mlp.gate` + `e_score_correction_bias` + MLA, **no indexer** (S, reuses layer 0).
- 1 NextN block (layer 2): sparse-like + `shared_head.norm`.
- Root: `token_embd`, `output_norm`, `output`.

Assertions:
- Output HF tensor name set matches expected exactly (no `blk.*` leakage, no orphan S-layer indexers).
- `build_config` output matches GLM-5.2 keys/values at structural level.
- F/S invariant holds.

---

## 6. Recommended execution order (dependency-driven)

1. **2.1** — arch-dispatch refactor (unblocks all MLA work; standalone safe).
2. **2.9** — `build_config` glm-dsa branch (pure metadata → dict; easy win).
3. **2.6 + 2.4 + 2.5** — indexer, bias, shared-expert name rules (pure renames).
4. **2.8** — NextN block + block_count arithmetic.
5. **2.7** — IndexShare invariant assertion (after 2.6 lands).
6. **2.3** — per-expert expert layout (highest complexity; last, behind format-detection).
7. **2.2** — kv_b concat for deepseek (folded into convert loop; optional but recommended).
8. **2.10** — tokenizer/template.
9. **3.1 + 3.2** — synthetic fixture + regression.
10. README update — add `glm-dsa` to architecture table with experimental footnote.

---

## 7. Acceptance criteria

- `detect_architecture("glm-dsa")` → `glm_moe_dsa`; class resolves to `GlmMoeDsaForCausalLM`.
- `build_config` output key/value-matches `zai-org/GLM-5.2/config.json` for MLA + MoE + indexer + MTP + rope.
- Every GGUF tensor in §2.2 remaps to a HF name present in real `model.safetensors.index.json`; no `blk.*` leakage, no phantom S-layer indexers.
- Tokenizer + chat template reproduce GLM specials and multi-EOS.
- New unit tests + smoke pass; no regression for existing archs (deepseek2, qwen2moe, qwen3moe, llama).

---

## 8. External blockers (unchanged)

1. **No GGUF producer in llama.cpp master** — `conversion/glm.py` registers `Glm4*` but not `GlmMoeDsa*`. Schema + tensor enum + metadata keys exist (small upstream PR away). Until then test input = synthetic fixture or community fork.
2. **No `mlx-lm` model class** — `mlx_lm/models/` has no glm family. Converted MLX dir loads only once `glm_moe_dsa.py` ships upstream (DSA indexer kernel is the hard part there — inference concern, not conversion concern).

The conversion work itself is fully specified and low-risk — pure mechanical translator, every name confirmed against the published 1.5 TB weights.

---

## 9. Open decisions (need user input before/at implementation)

1. **Scope of MLA kv_b fix (Issue 2.2):** glm-dsa-only (trivial, combined) **or** also fix deepseek2/3 split-kv_b-concat (**recommended**, shared MLA family)?
2. **Expert target layout (Issue 2.3):** per-expert `experts.{e}` (matches HF GLM-5.2 reference — **recommended**) or keep mlx `switch_mlp` stacked (matches existing repo pattern but diverges from reference)?
3. **Proceed to implementation** now (switch to implement mode) or refine further first?

**Author recommendation:** (1) also fix deepseek, (2) per-expert, (3) proceed.

---

## 10. Research sources (all indexed, searchable)

| # | Source | arxiv/URL | Purpose |
|---|--------|-----------|---------|
| 1 | DeepSeek-V3.2 (DSA origin) | 2512.02556 | First DSA formalization, lightning indexer equation |
| 2 | StreamIndex (V4 CSA formalization) | 2605.02568 | V4 CSA pipeline: compressor, indexer math, dims |
| 3 | FlashMemory-DeepSeek-V4 | 2606.09079 | V4 hybrid HCA+CSA, 128:1 HCA compression |
| 4 | GLM-5 technical report | 2602.15763 | GLM-5 architecture: MLA+DSA+MoE+MTP |
| 5 | GLM-5.2 launch blog | HF blog (zai-org/glm-52-blog) | IndexShare + KVShare contributions |
| 6 | IndexCache (IndexShare paper) | 2603.12201 | F/S layer pattern, 1/4 retention, weight layout |
| 7 | MISA (3rd-party DSA repro) | 2605.07363 | Independent DSA indexer re-impl cross-check |
| 8 | GLM-5.2 safetensors index | huggingface.co/zai-org/GLM-5.2 | Ground-truth HF tensor names |
| 9 | GLM-5.2 config.json | huggingface.co/zai-org/GLM-5.2 | Ground-truth config values |
| 10 | llama.cpp gguf constants | raw.githubusercontent.com/.../constants.py | GGUF glm-dsa tensor enum + metadata keys |
