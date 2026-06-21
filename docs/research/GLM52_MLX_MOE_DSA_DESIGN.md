# Design: `glm_moe_dsa.py` MLX-lm model file (GLM-5.2)

**Status:** DESIGN (no code yet). Companion to `GLM52_FORWARD_PATH_REFERENCE.md`.
**Target file:** `.venv-mlx/lib/python3.12/site-packages/mlx_lm/models/glm_moe_dsa.py`
(replaces the current 53-line no-op `Model(DSV32Model)` subclass).
**Scope:** SHORT CONTEXT ONLY (L < index_topk=2048 → indexer returns None → standard MLA path).
The IndexShare / F-S forward path is explicitly OUT OF SCOPE (separate design).

---

## TL;DR (the headline finding)

**The forward graph in `deepseek_v32.py` is already correct for GLM-5.2 — except ONE divergence.**

The session-memory prime suspect ("RoPE interleave mismatch → flip `traditional`") is **wrong**.
MLX's `traditional=True` already produces the interleaved/adjacent-pairs rotation that
`rope_interleave=True` requests (verified against `mlx.nn.RoPE` docstring + `gpt_neox.py`
+ `nemotron-nas.py`). Flipping it would **break** the model.

The single confirmed forward-path divergence is **MODERATE**, not CRITICAL:

| Where | `deepseek_v32.py` | GLM-5.2 needs | Severity |
|---|---|---|---|
| L136 `q_a_layernorm` | `nn.RMSNorm(q_lora_rank, eps=1e-6)` | `eps=config.rms_norm_eps` (≈1e-5) | MODERATE |
| L146 `kv_a_layernorm` | `nn.RMSNorm(kv_lora_rank, eps=1e-6)` | `eps=config.rms_norm_eps` (≈1e-5) | MODERATE |

All other norms in the file (decoder L392/394, model L422) already read `config.rms_norm_eps`.
So the new file's **only** behavioral change is to fix these two norms.

**Honest caveat:** the eps fix alone is unlikely to restore coherent output. The
degenerate repetition observed in Story 7.M AC6 is most plausibly the **2-bit
quantization downgrade** (GGUF IQ2_S @ 2.64 bpw with importance matrix → naive MLX
affine 2-bit @ 2.0 bpw, no importance matrix, across all 189 expert modules). The
testing plan below isolates quantization from forward-path before any deeper
rewrites.

---

## 1. Class hierarchy

**Imports from `deepseek_v32`:**
- `Model as DSV32Model` (top-level causal LM — the thing `mlx_lm.load()` instantiates).
- `ModelArgs as DSV32ModelArgs` (NOT strictly needed; the existing `glm_moe_dsa.ModelArgs`
  is already correct — see §2).
- (reference only) `DeepseekV32Attention`, `DeepseekV32DecoderLayer`, `DeepseekV32Model`.

**Recommended minimal design** — one class, post-init norm patch:

```python
from mlx import nn
from .deepseek_v32 import Model as DSV32Model

class Model(DSV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        # GLM-5.2 fix: MLA-internal norms must use config eps.
        # deepseek_v32.py:136,146 hardcode eps=1e-6; GLM-5.2 rms_norm_eps≈1e-5.
        eps = config.rms_norm_eps
        for layer in self.model.layers:
            sa = layer.self_attn
            sa.q_a_layernorm  = nn.RMSNorm(sa.q_lora_rank,  eps=eps)
            sa.kv_a_layernorm = nn.RMSNorm(sa.kv_lora_rank, eps=eps)
```

**Why this works:** `mlx.nn.Module.__setattr__` re-binds submodules by name, and weights
load *after* `__init__` (via `load_weights`/`sanitize`), binding by dotted path
(`model.layers.N.self_attn.q_a_layernorm.weight`). The re-assignment is idiomatic and
the named weights bind to the new RMSNorm modules cleanly.

**Extensible alternative** (preferred once IndexShare work begins — gives a natural
home for the F/S forward path):

```python
class GlmMoeDsaAttention(DeepseekV32Attention):     # override __init__ ONLY
    def __init__(self, config):
        super().__init__(config)
        self.q_a_layernorm  = nn.RMSNorm(self.q_lora_rank,  eps=config.rms_norm_eps)
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

class GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = GlmMoeDsaAttention(config)

class GlmMoeDsaModel(DeepseekV32Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = [GlmMoeDsaDecoderLayer(config, i)
                       for i in range(config.num_hidden_layers)]

class Model(DSV32Model):
    def __init__(self, config):
        super().__init__(config)
        self.model = GlmMoeDsaModel(config)
```

(Trade-off: idiomatic + extensible, but doubles layer-object allocation during
`__init__` — harmless, no weights are loaded yet, ~ms.) Choose minimal for the
first landing; switch to the chain when IndexShare lands.

---

## 2. ModelArgs changes

The existing `glm_moe_dsa.ModelArgs` already loads the config cleanly (verified: load
succeeds in 28s, AC5 PASS). Changes below are **OPTIONAL / traceability only**:

| Change | Purpose | Behavioral effect |
|---|---|---|
| ADD `rope_interleave: bool = True` | Stop `config.json`'s `rope_interleave` being silently dropped; assert `== True` in `__post_init__` (RoPE uses `traditional=True` unconditionally) | None (assert-only) |
| ADD `expert_gating_func: Optional[int] = None` | Provenance for the GGUF field dropped by the converter; assert `None or == 2` in `__post_init__` | None (gate uses `scoring_func`) |
| KEEP `__post_init__` (rope_parameters → rope_scaling/rope_theta) | — | unchanged |

Keep `ModelArgs` field set otherwise identical to today. Do NOT remove any field.

---

## 3. RoPE fix

**NO CHANGE.** `deepseek_v32.py:172` (attention) and `:74` (indexer) call
`initialize_rope(dims=qk_rope_head_dim=64, base=rope_theta=8e6, traditional=True,
scaling_config=rope_scaling=None)`. All four parameters are correct for GLM-5.2:

- `traditional=True` ⟺ **interleaved / adjacent-pairs** (GPT-J) ⟺ exactly what HF
  `rope_interleave=True` requests. (Cross-check: `gpt_neox.py:48` uses
  `traditional=False` for NeoX split-half; `nemotron-nas.py:177` comment confirms
  Llama = `traditional=False`.) Flipping to `False` would **scramble** positions.
- `dims=64` matches GGUF `glm-dsa.rope.dimension_count=64`. `partial_rotary_factor=0.25`
  is redundant metadata (0.25 × qk_head_dim 256 = 64 = qk_rope_head_dim); correctly ignored.
- `base=8e6` flows from config.
- `rope_scaling=None` → the mscale block (`deepseek_v32.py:160-166`) is correctly inert.

---

## 4. Attention forward changes

**NO `__call__` override.** The forward math (`deepseek_v32.py:177-261`) is correct for
GLM-5.2's MLA-256 layout: `q_a_proj(6144→2048) → q_a_layernorm → q_b_proj(2048→16384)`,
`kv_a_proj_with_mqa(6144→576) → split{512,64} → kv_a_layernorm(512) → kv_b via
embed_q/unembed_out`, decoupled RoPE on `q_pe/k_pe`, absorbed-MLA attention. Every
dimension flows from config (`v_head_dim=256`, `q_lora_rank=2048`, `q_head_dim=256`,
`num_heads=64`). Verified consistent with the weight format on disk.

**ONLY init-level change:** re-create `q_a_layernorm` + `kv_a_layernorm` with
`eps=config.rms_norm_eps` (see §1). This is the single confirmed forward-path divergence.

---

## 5. MoE gate changes

**NO CHANGE.** `deepseek_v32.MoEGate` (`deepseek_v32.py:317-340`) + `group_expert_select`
already match GLM-5.2:

- `scores = mx.sigmoid(...)` (L293) ⟺ `scoring_func: sigmoid` (config) ⟺
  `expert_gating_func: 2` (llama.cpp enum, ABSENT from config.json — redundant).
- `assert config.topk_method == "noaux_tc"` (L329) ⟺ `topk_method: noaux_tc` (config).
- `n_group=1, topk_group=1` → group-filter block is a no-op (matches config).
- `norm_topk_prob=True`, `routed_scaling_factor=2.5`, `num_experts_per_tok=8` all read
  from config.

Do **not** branch on `expert_gating_func`. Optional: add an assert in `ModelArgs.__post_init__`
that `scoring_func == "sigmoid"` to fail loud on a future contradicting config.

---

## 6. sanitize() override

**NO OVERRIDE.** `deepseek_v32.Model.sanitize` (L494-503) strips MTP layers
(`parts[2] >= num_hidden_layers` → `continue`), already correct for 66 layers. The
per-tensor weight format on disk is verified correct (Story 7.M AC1-AC5 PASS):
`kv_b_proj` combined `[k_b_t, v_b]`, `embed_q`/`unembed_out` split-by-heads MLA weights,
experts stacked via `SwitchGLU`. The earlier session bugs (A: v_head_dim misread, B:
rope_parameters nesting, C: `exp_probs_b.bias → e_score_correction_bias`, D:
`experts.N.* → switch_mlp.*` quant-collapse) are already fixed in the exported shards.
Inherit `sanitize` as-is.

---

## 7. Testing plan

Three tiers, in priority order.

### Tier 1 — Isolation experiment (HIGHEST LEVERAGE, run FIRST)
Determines whether the forward path is even the issue, OR the 2-bit quant downgrade is.

- Re-export routed experts at **4-bit affine** (instead of 2-bit) for a small slice
  — either re-export only layers 0-5, or a tiny 2-layer test model — into a throwaway
  `-mlx-4bit-probe` folder.
- Run the merge-sort prompt + the trivial prompts (Tier 3) on the probe.
- **Decision:**
  - coherent → cause = 2-bit quant downgrade. Forward path vindicated; close this design.
    Re-plan PLAN §7.M AC6 to use ≥3-bit experts.
  - still degenerate → forward-path bug remains. Escalate to the IndexShare KV-cache
    coupling audit (see Unknowns #5 / §8 minimal-viable step 3) and re-read the
    GLM-5 tech report RMSNorm/scale definitions.

### Tier 2 — eps-fix validation
After shipping the §1 fix on the FULL model, re-run the same prompts. Expect small
improvement; if collapse persists it confirms quantization is the dominant cause.

### Tier 3 — Non-degeneracy smoke test (trivial prompts)
Run BEFORE and AFTER each change so the cause is attributable:
- "What is 2+2?" → expect "4"
- "Say hello." → expect a greeting
- "Complete: The sky is" → expect "blue"
Each must produce coherent, non-repeating output within 40 tokens.

### Tier 4 — Merge-sort baseline (canonical)
Commit a clean MLX script mirroring `scripts/baselines/glm52_merge_sort_baseline.sh`,
e.g. `scripts/baselines/glm52_mlx_merge_sort_baseline.sh`: load MLX model, apply chat
template, stream 512 tokens, print tok/s + full output, sanity-test the generated sort
on 6 cases. **Pass** = a correct merge sort that sorts a sample array.

**Acceptance gate for this design:** Tier 3 trivial prompts produce coherent output on
the 4-bit probe (Tier 1). Merge-sort correctness (Tier 4) is the stretch goal.

---

## 8. Minimal viable forward

Smallest set of changes that could restore coherent output — **incremental, do not
rewrite everything**:

1. **Ship §1** — `Model(DSV32Model)` overriding `__init__` with the 4-line norm-eps
   patch loop. Fixes the only confirmed divergence. (~10 LOC total file size.)
2. **Run Tier 3** (trivial prompts) immediately on the full 2-bit model.
3. **If still degenerate → run Tier 1** (4-bit-experts isolation re-export). This is
   the actual highest-leverage fix per Deliverable 1.
4. **If Tier 1 also degenerate →** short-circuit the `Indexer` + the
   `cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))` line
   (attention `__call__` tail) for L<index_topk and re-test, to rule out indexer-cache
   coupling silently perturbing the MLA KV path at short context.

Do NOT attempt IndexShare / long-context in this pass.

---

## 9. Unknowns (require deeper PDF reading or empirical resolution)

1. **Attention scale denominator.** `deepseek_v32.py:131` uses `q_head_dim**-0.5 =
   256**-0.5` (qk_nope 192 + qk_rope 64). INFERRED correct (matches config
   `qk_head_dim: 256`), but not yet cited from the GLM-5 tech report. If the report
   specifies `head_dim: 192` or a per-head scale, logits rescale by √(256/192)≈1.155.
   Low-impact; confirm via report §attention.

2. **RMSNorm `+1` quirk.** Some GLM variants use `x / sqrt(mean(x²) + 1)` instead of
   `+ eps`. `mlx.nn.RMSNorm` uses `+ eps` (standard). INFERRED standard (no source
   mentions `+1` for GLM-5.2). If the report specifies `+1`, a custom RMSNorm is
   required. Confirm via report §RMSNorm.

3. **`expert_gating_func` future variants.** Resolved for THIS config (=sigmoid=2).
   If a future GLM-5.2 config ships `expert_gating_func != 2`, `deepseek_v32`'s
   hardcoded sigmoid would be silently wrong. Mitigation: assert in `__post_init__`.

4. **2-bit affine quantization floor.** UNKNOWN whether MLX 2-bit affine (no importance
   matrix) is ever sufficient for a 256-expert MoE at short context. Tier 1 answers
   this empirically. If insufficient, PLAN §7.M AC6 needs ≥3-bit experts.

5. **Indexer-cache coupling at short context.** Although the indexer returns None for
   L=28 < 2048, the `mx.depends(...)` line coupling `cache[0]` to `cache[1]` runs
   unconditionally. Empirically verify (minimal-viable step 4) it is not silently
   perturbing the MLA KV path before investing in a full IndexShare implementation.

6. **IndexShare F/S forward path.** OUT OF SCOPE here. The documented long-context
   blocker (PLAN §8 item 2; `GLM52_FORWARD_PATH_REFERENCE.md` §indexer). Gets its own
   design after this one lands and short-context is verified.
