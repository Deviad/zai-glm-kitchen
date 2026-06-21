# GLM-5.2 Forward-Path Architecture Reference

**Purpose.** Ground-truth reference for implementing a correct GLM-5.2 forward
pass in `mlx-lm` (and equivalently in `llama.cpp`'s `glm-dsa.cpp`). Every fact
is tagged **CERTAIN** (verbatim from config.json, the GLM-5 tech report, the
IndexCache paper, or the DeepSeek-V3.2 DSA paper) or **INFERRED** (canonical
MLA/DeepSeek convention confirmed by tensor-name shape, not stated verbatim).

**Sources cited.**
- `[CFG]` = `/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-shortgpt-pruned-mixed-mlx/config.json` (actual pruned MLX config; all top-level non-quantization fields read & listed in §7)
- `[TR]`  = GLM-5 tech report, `docs/research/papers/glm5-tech-report-2602.15763.pdf`
- `[IC]`  = IndexCache / IndexShare paper, `docs/research/papers/indexcache-indexshare-2603.12201.pdf`
- `[DS32]` = DeepSeek-V3.2 paper (DSA origin), `docs/research/papers/deepseek-v3.2-dsa-2512.02556.pdf`
- `[PLAN]` = `PLAN.md` §2 ground-truth (HF tensor names verified against the 1.5 TB GLM-5.2 weights index)

---

## ⚠️ Correction to the task brief

The brief assumed `expert_gating_func=2` is a config field. **It is NOT present
in this config.** The full top-level (non-quantization) key list is in §7 below
and contains no `expert_gating_func` / `expert_gating` field of any kind
(verified by grepping the serialized config). GLM-5.2's MoE gating is encoded
by the pair `scoring_func="sigmoid"` + `topk_method="noaux_tc"` — the same pair
DeepSeek-V3 uses. `expert_gating_func` is a field introduced by a *different*
GLM release (GLM-4.6 / newer transformers builds) and does not apply here.

---

## 1. MLA (Multi-head Latent Attention) layout

### 1.1 Dims — all CERTAIN from `[CFG]`

| Field | Value | Note |
|---|---|---|
| `num_attention_heads` | **64** | `[CFG]` |
| `q_lora_rank` | **2048** | `[CFG]` |
| `kv_lora_rank` | **512** | `[CFG]` |
| `qk_nope_head_dim` | **192** | `[CFG]` |
| `qk_rope_head_dim` | **64** | `[CFG]` |
| `qk_head_dim` | **256** | `[CFG]` = `qk_nope_head_dim + qk_rope_head_dim` (192+64) |
| `v_head_dim` | **256** | `[CFG]` |
| `head_dim` | **192** | `[CFG]` — legacy field; equals `qk_nope_head_dim`, NOT the effective 256 |
| `num_key_value_heads` | **1** | `[CFG]` — MQA-style single KV head (DSA runs under MLA's MQA mode) |
| `hidden_size` | **6144** | `[CFG]` |

**Why these dims (CERTAIN rationale, `[TR]` §2.1, lines 213–220).** The GLM-5
report describes an **"MLA-256"** variant: *"we increase the head dimension
from 192 to 256 and decrease the number of attention heads by 1/3 … while
decreasing the decoding computation. The variant, denoted as MLA-256, matches
the performance of MLA under Muon Split."* The config's `qk_head_dim=256` /
`v_head_dim=256` is exactly this MLA-256. (GLM-5.2's head count of 64 is
config-authoritative; it is *not* a clean 2/3 of DeepSeek-V3's 128 — GLM-5.2
chose 64 independently, so treat 64 as `[CFG]` ground truth, not derivable
from the report's "1/3" sentence.)

### 1.2 The q_a_proj → q_a_norm → q_b_proj compression flow — INFERRED (canonical MLA, confirmed by `[PLAN]` §2.1 tensor names)

Tensor names present in HF weights (`[PLAN]` §2.1): `self_attn.q_a_proj`,
`self_attn.q_a_layernorm`, `self_attn.q_b_proj` (all `.weight`).
The flow is the standard DeepSeek MLA query path:

```
h ∈ R[B, L, 6144]
  ──q_a_proj──▶  c_q ∈ R[B, L, 2048]      # down-compress to q_lora_rank
  ──q_a_layernorm (RMSNorm)──▶            # RMSNorm on the 2048 latent
  ──q_b_proj──▶  q ∈ R[B, L, 64·256=16384]  # up-project to heads×qk_head_dim
                  reshape → [B, 64, L, 256]
                  split last dim → q_nope ∈ [B,64,L,192] , q_pe ∈ [B,64,L,64]
```

`q_b_proj` output width = `num_attention_heads × qk_head_dim` = `64 × 256` =
**16384**. The first 192 dims per head are `q_nope`; the last 64 are `q_pe`
(the RoPE carrier). The split order (nope-then-rope within each head) is the
DeepSeek convention; **INFERRED** — confirm against `mlx_lm.models.deepseek_v3`
before shipping.

### 1.3 The kv_a_proj_with_mqa → kv_a_norm → kv_b_proj flow — INFERRED (canonical MLA + MQA), confirmed by `[PLAN]` §2.1 + `[DS32]` L162–164

Tensor names (`[PLAN]` §2.1): `self_attn.kv_a_proj_with_mqa`,
`self_attn.kv_a_layernorm`, `self_attn.kv_b_proj`. DeepSeek-V3.2 confirms DSA
runs *"under the MQA mode of MLA, where each latent vector … will be shared
across all query heads"* (`[DS32]` L162–164). Flow:

```
h ∈ R[B, L, 6144]
  ──kv_a_proj_with_mqa──▶  c_kv_pe ∈ R[B, L, 576]   # = kv_lora_rank + qk_rope_head_dim = 512+64
        split last dim → c_kv ∈ [B,L,512] , k_pe ∈ [B,L,64]
  ──kv_a_layernorm (RMSNorm on the 512 part ONLY; k_pe is NOT normed)──▶
  ──kv_b_proj──▶  kv ∈ R[B, L, 64·(192+256)=28672]   # 64 heads × (qk_nope + v_head)
                  reshape → [B, 64, L, 448]
                  split → k_nope ∈ [B,64,L,192] , v ∈ [B,64,L,256]
  Final key per head: concatenate([k_nope, k_pe], dim=-1) → [B,64,L,256]
                       (k_pe is broadcast across the 64 heads)
```

- `kv_a_proj_with_mqa` output width = `kv_lora_rank + qk_rope_head_dim` =
  `512 + 64` = **576**. CERTAIN shape (DeepSeek MLA invariant).
- `kv_b_proj` output width = `num_attention_heads × (qk_nope_head_dim + v_head_dim)`
  = `64 × (192 + 256)` = **28672**. CERTAIN shape.
- **`kv_b_proj` is a SINGLE combined projection** — GLM-5.2 HF does **not**
  split into `k_b` / `v_b` (`[PLAN]` §2.1 note). This is the same combined
  layout as DeepSeek-V3 HF. There is **no** `k_nope_head_norm` /
  `k_rope_head_norm` in GLM-5.2 (`[PLAN]` §2.1 note) — unlike some DeepSeek
  reimplementations that add them. **DIVERGENCE FLAG**: do not add extra norms.

### 1.4 Where k_pe (the RoPE carrier) lives — INFERRED (canonical DeepSeek decoupled RoPE)

`k_pe` comes out of `kv_a_proj_with_mqa` (the trailing 64 dims of the 576
output). It is shared across all 64 query heads (MQA). `q_pe` comes out of
`q_b_proj` (the trailing 64 dims per head). Both are rotated by RoPE, then
`k_pe` is concatenated per-head with `k_nope` to form the full 256-dim key.
This is the **decoupled RoPE** scheme — see §2.

---

## 2. RoPE convention

| Field | Value | Certainty |
|---|---|---|
| `rope_theta` | **8,000,000.0** (8e6) | `[CFG]` CERTAIN |
| `rope_scaling` | **None** | `[CFG]` CERTAIN (`rope_parameters.rope_scaling = None`, `rope_type = "default"`) |
| `rope_interleave` | **True** | `[CFG]` CERTAIN |
| `partial_rotary_factor` | **0.25** | `[CFG]` CERTAIN |
| Max position | 1,048,576 | `[CFG]` `max_position_embeddings` |

### 2.1 Decoupled RoPE — CERTAIN scheme, INFERRED mechanical detail

GLM-5.2 uses **DeepSeek's decoupled RoPE**: a dedicated `qk_rope_head_dim=64`
slice (`q_pe` / `k_pe`) is rotated separately from the `qk_nope_head_dim=192`
slice, then concatenated for the attention dot product. The `[TR]` report does
not spell this out, but the config's split dims (`qk_nope_head_dim` /
`qk_rope_head_dim`) and the `[DS32]` figure 2 description (the `[q^A; q^R_t,i]`
concatenation, `[c^KV ; k^R_t]` concatenation) are unambiguous. Same scheme as
DeepSeek-V3.2 — **NO divergence here**.

`partial_rotary_factor=0.25` = `qk_rope_head_dim / qk_head_dim` = `64/256` =
0.25. CERTAIN consistency check; it confirms only the trailing 64 of each
256-dim head is rotated.

### 2.2 What `rope_interleave=True` means — INFERRED (framework convention; verify against `mlx_lm`'s rope impl)

- **`interleave=True`** (GPT-J / "adjacent pairs" style): rotation pairs
  dimensions `(2i, 2i+1)`. I.e. for a 64-dim rope head, pairs are
  `(d0,d1), (d2,d3), …, (d62,d63)`.
- **`interleave=False`** (GPT-NeoX / "split-half" style): rotation pairs the
  first half against the second half: `(d_i, d_{i+32})` for a 64-dim head.

DeepSeek-V3 also ships `rope_interleave=True` in HF, so GLM-5.2 matches
DeepSeek here. **Action item:** confirm `mlx_lm`'s rope helper honors an
`interleave` flag at all (stock HF Llama-style rope is split-half only).
This is a likely silent-bug source if the MLX port hard-codes split-half.

### 2.3 rope_scaling = None — CERTAIN

No YaRN, no NTK, no linear/dynamic scaling. Plain RoPE at `theta=8e6`. **DIVERGENCE FLAG**: DeepSeek-V3.2
HF ships a `rope_scaling` config (YaRN) for its long-context variant; GLM-5.2
ships **None** and relies on DSA + the raw 8e6 theta. Do not import a YaRN
config from a DeepSeek reference implementation.

---

## 3. RMSNorm

| Field | Value | Certainty |
|---|---|---|
| `rms_norm_eps` | **9.999999747378752e-06** ≈ 1e-5 | `[CFG]` CERTAIN |
| `+1` in denominator? | **NO** (standard RMSNorm) | INFERRED |

**`+1` quirk:** neither `[TR]` nor `[CFG]` mention a `+1` denominator variant.
Some older ChatGLM lineages used `RMSNorm(x, w) = x / sqrt(mean(x²) + eps) * w`
— i.e. **standard** with eps inside the sqrt. There is no evidence GLM-5.2
switches to `x / (mean(x²)+1) * w` or any other `+1` form. Treat as
**standard RMSNorm with eps inside the sqrt**, applied to:
`input_layernorm`, `post_attention_layernorm`, `q_a_layernorm`,
`kv_a_layernorm`, `model.norm`, and the MTP `shared_head.norm`. **Mark this
INFERRED** — confirm against the GLM-5.2 HF model code if it ever lands in
transformers, but do not invent a `+1`.

---

## 4. MoE gate

### 4.1 Config fields — all CERTAIN from `[CFG]`

| Field | Value |
|---|---|
| `scoring_func` | **"sigmoid"** |
| `topk_method` | **"noaux_tc"** |
| `norm_topk_prob` | **True** |
| `routed_scaling_factor` | **2.5** |
| `n_group` | **1** |
| `topk_group` | **1** |
| `num_experts` / `n_routed_experts` | **256** |
| `num_experts_per_tok` | **8** |
| `n_shared_experts` | **1** |
| `moe_intermediate_size` | **2048** |
| `first_k_dense_replace` | **3** (layers 0–2 dense, 3+ sparse) |
| `moe_layer_freq` | 1 |
| `decoder_sparse_step` | 1 |

### 4.2 What the gate actually computes — INFERRED (DeepSeek-V3 convention; identical field set)

With `n_group=1, topk_group=1`, the group-filtering stage is a **no-op**
(single group contains all 256 experts). The effective gate is:

```
scores = sigmoid(gate_proj(h))                  # sigmoid, NOT softmax — scoring_func="sigmoid"
# noaux_tc bias-corrected top-8 selection:
scores_for_rank = scores - e_score_correction_bias   # mlp.gate.e_score_correction_bias, [256]
topk_idx  = top-8(scores_for_rank)               # noaux_tc = "no auxiliary loss, top-k with correction"
topk_w    = gather(scores, topk_idx)             # the RAW sigmoid scores at the chosen experts
if norm_topk_prob:  topk_w = topk_w / sum(topk_w)  # norm_topk_prob=True → renormalize
topk_w    = topk_w * routed_scaling_factor       # ×2.5
output    = Σ_e topk_w[e] · expert_e(h)  +  shared_expert(h)   # 1 shared expert, always on
```

- **`scoring_func="sigmoid"`** → per-expert score is `sigmoid(logit)`, not a
  softmax over experts. Each expert is scored independently. This is the
  DeepSeek-V3 / V3.2 convention.
- **`topk_method="noaux_tc"`** = "no auxiliary loss, top-k with correction
  bias" — the DeepSeek-V3 aux-loss-free routing. `e_score_correction_bias`
  (GGUF: `exp_probs_b`, HF: `mlp.gate.e_score_correction_bias`) is a learned
  per-expert bias subtracted before the top-k argmax. Confirmed tensor present
  in `[PLAN]` §2.1 and §2.2.
- **`expert_gating_func=2`**: **NOT PRESENT in this config** — see top of
  file. Do not branch on it.
- **`norm_topk_prob=True` + `routed_scaling_factor=2.5`**: renormalize the
  8 selected sigmoid scores to sum to 1, then multiply by 2.5.

This is **identical to DeepSeek-V3/V3.2**'s MoE gate — **NO divergence** in
the *mechanism*. The divergence is in the *counts*: 256 experts × 2048 hidden
× 8 active × 1 shared, vs DeepSeek-V3's 256 × 2048 × 8 × 1 (actually very
close — GLM-5.2 matches DS-V3's MoE sizing almost exactly).

### 4.3 SwiGLU activation — CERTAIN (`hidden_act`) + INFERRED (GLU wiring)

`hidden_act = "silu"` (`[CFG]`). With the `gate_proj` / `up_proj` /
`down_proj` trio (confirmed in `[PLAN]` §2.1), the expert FFN is standard
**SwiGLU**: `down_proj(silu(gate_proj(h)) * up_proj(h))`. Same for the dense
layers (0–2) and the single shared expert. No quirk.

---

## 5. Indexer (DSA) — for completeness; NOT the short-context bug

### 5.1 Config dims — all CERTAIN from `[CFG]`

| Field | Value |
|---|---|
| `index_n_heads` | **32** (= H_I, the indexer head count) |
| `index_head_dim` | **128** (= d_I, per-head indexer dim) |
| `index_topk` | **2048** (k) |
| `index_share_for_mtp_iteration` | **False** |

Indexer tensors present on **F (Full) layers only** (`[PLAN]` §2.1):
`self_attn.indexer.{wq_b, wk, weights_proj}.weight`,
`self_attn.indexer.k_norm.{weight,bias}`. GGUF names (`[PLAN]` §2.2):
`indexer.attn_q_b`, `indexer.attn_k`, `indexer.proj`, `indexer.k_norm`.

### 5.2 The lightning-indexer scoring equation — CERTAIN math from `[DS32]` eq. (1), L130–150

```
I_{t,s} = Σ_{j=1..H_I}  w^I_{t,j} · ReLU( q^I_{t,j} · k^I_s )      # [DS32] eq.(1)
        = Σ_{j=1..32}   w_{t,j}  · ReLU( q_{t,j} · k_s )            # dims: q,k ∈ R^128, w ∈ R
```

- `H_I = 32` (config `index_n_heads`), `d_I = 128` (config `index_head_dim`).
- **Activation = ReLU** (not sigmoid/softmax) — `[DS32]` L145 "We choose ReLU
  as the activation function for throughput consideration." Also `[TR]` L1264:
  *"Lightning Indexer integrates score calculation, ReLU, and TopK operations
  into a single kernel."*
- `q^I_{t,j}` and `w^I_{t,j}` are derived from the query token `h_t`;
  `k^I_s` is derived from the preceding token `h_s`. Tensor mapping
  (`[PLAN]` §2.1): `wq_b` produces `(q^I, w^I)`, `wk` produces `k^I`,
  `weights_proj` produces the per-head scalar weights `w^I`, `k_norm` is a
  norm on `k^I` (has both `.weight` and `.bias`, so it is an affine norm, not
  a bare RMSNorm — **DIVERGENCE FLAG**: this `k_norm` with bias is not the
  same as the bias-free `q_a_layernorm`/`kv_a_layernorm` RMSNorms).
- **Top-k = 2048 per query token** (`[DS32]` L246: "select 2048 key-value
  tokens for each query token"; config `index_topk=2048`). `[TR]` L673 confirms
  `k = 2048` for GLM-5's indexer.
- After top-k selection, attention `u_t = Attn(h_t, {c_s : I_{t,s} ∈ Top-k})`
  (`[DS32]` eq. 2) runs sparsely over the 2048 retrieved KV entries.

### 5.3 When does the indexer activate? — INFERRED (natural DSA behavior)

The indexer is a **no-op when L ≤ k**: if the prefix is ≤ 2048 tokens,
top-2048-of-L selects all L tokens, so DSA degenerates to dense MLA. This is
the standard DSA short-circuit and is consistent with the `[TR]`/`[DS32]`
descriptions. **The REAP37 short-context merge-sort passing is expected** —
the indexer simply doesn't fire. **Any short-context (L ≤ 2048) bug is NOT an
indexer bug**; it must live in MLA / RoPE / RMSNorm / MoE. The REAP37
long-context gibberish at 4.9k and 18.7k *is* an indexer-path failure (the
compat hack duplicates F-layer indexers into S layers, which is semantically
wrong — see §6).

### 5.4 The F/S (Full/Shared) IndexShare pattern — CERTAIN from `[IC]` + `[PLAN]`

`[IC]` abstract + L112–115: *"IndexCache partitions layers into **F (Full)
layers** that retain their indexers and **S (Shared) layers** that inherit
top-k indices from the nearest preceding F layer, adding only one conditional
branch in inference."* The greedy training-free solution *"retains only 1/4
of indexers"* (`[IC]` L121). `[PLAN]` §2.4 confirms GLM-5.2 uses 1/4 retention
and that the per-layer role is encoded in `indexer_types[]` (config — though
the field is not exposed at top level in *this* pruned config; it is in the
full GLM-5.2 config per `[PLAN]` §2.3). Confirmed in weights: layer 1 has
`indexer.*`, layer 27 has none.

**Forward-path implication (the actual blocker, `[PLAN]` §8 item 2):** an S
layer has **no indexer tensors** and must reuse the **top-k index set**
computed by the nearest preceding F layer — it does NOT skip attention, only
the indexer scoring. `mlx_lm/models/glm_moe_dsa.py` subclasses
`deepseek_v32.Model` which has zero IndexShare logic; S layers crash on
missing `indexer.*` params. The REAP37 "compat hack" (duplicate F-layer
indexer weights into S layers) makes it load but produces gibberish at long
context because S layers are then scoring with the wrong indexer weights
instead of reusing F's index set.

### 5.5 MTP / index interaction

`index_share_for_mtp_iteration = False` (`[CFG]`): the MTP (NextN) layer does
**not** share the indexer iteration's top-k. `num_nextn_predict_layers = 1`
(`[CFG]`): one MTP layer (block 66 in this pruned config = block 78 in full
GLM-5.2). The MTP layer carries an extra `shared_head.norm.weight`
(`[PLAN]` §2.1/§2.2). `[TR]` §2.1 (L222–235): GLM-5 trains with **3 MTP layers
sharing parameters**, predicts next-2 tokens at inference, accept length 2.76
vs DS-V3.2's 2.55. For a single-token-decode forward path, the MTP layer is
not exercised; it only matters for speculative decoding.

---

## 6. Divergence summary vs DeepSeek-V3.2 defaults

| Aspect | DeepSeek-V3.2 default | GLM-5.2 | Divergence? |
|---|---|---|---|
| MLA scheme | standard MLA | MLA-256 variant | **YES** — head dim 192→256, v_head 128→256 |
| `q_lora_rank` | 1536 | **2048** | **YES** — larger query latent |
| `kv_lora_rank` | 512 | 512 | no |
| `qk_nope_head_dim` | 128 | **192** | **YES** |
| `qk_rope_head_dim` | 64 | 64 | no |
| `v_head_dim` | 128 | **256** | **YES** |
| `num_attention_heads` | 128 | **64** | **YES** |
| Decoupled RoPE | yes | yes | no |
| `rope_interleave` | True | True | no (but MLX port must honor it) |
| `rope_theta` | 1e4 (V3 base) | **8e6** | **YES** |
| `rope_scaling` | YaRN (long-ctx) | **None** | **YES** — DSA replaces YaRN |
| RMSNorm `+1` | no | no (inferred) | no |
| MoE `scoring_func` | sigmoid | sigmoid | no |
| MoE `topk_method` | noaux_tc | noaux_tc | no |
| MoE counts | 256 / 8 / 1 shared, 2048 hidden | 256 / 8 / 1 shared, 2048 hidden | no (nearly identical) |
| `routed_scaling_factor` | 2.5 (V3) | 2.5 | no |
| SwiGLU | yes | yes | no |
| Indexer per layer | every layer | **F/S IndexShare, 1/4 F** | **YES** — GLM-5.2-specific novelty (`[IC]`) |
| `index_topk` | 2048 | 2048 | no |
| Indexer `k_norm` | — | **has `.bias`** (affine) | **YES** — not a bare RMSNorm |
| MTP layers | 1 | 1 (`[CFG]`); trains with 3 shared (`[TR]`) | no (decode path) |

**The four implementation-critical divergences** (any one silently breaks the
forward pass): (a) MLA-256 head dims, (b) `rope_theta=8e6` + `rope_scaling=None`,
(c) IndexShare F/S forward path, (d) affine (bias) `k_norm` in the indexer.

---

## 7. Appendix — full top-level config (non-quantization), verbatim `[CFG]`

```
architectures: ['GlmMoeDsaForCausalLM']
model_type: glm_moe_dsa
hidden_size: 6144
intermediate_size: 12288
num_hidden_layers: 66                       # pruned; full GLM-5.2 = 78 (66 decoder + 1 MTP relative to blk.count=67)
num_attention_heads: 64
num_key_value_heads: 1
max_position_embeddings: 1048576
rms_norm_eps: 9.999999747378752e-06         # ≈ 1e-5
rope_theta: 8000000.0
vocab_size: 154880                          # note: full GLM-5.2 ≈ 151k; this pruned variant 154880
hidden_act: silu
tie_word_embeddings: False
attention_bias: False
torch_dtype: float16
bos_token_id: 154822
eos_token_id: 154820
_gguf_architecture: glm-dsa
_gguf_file_type: 25
_original_name: Glm-5.2
num_nextn_predict_layers: 1
q_lora_rank: 2048
kv_lora_rank: 512
qk_nope_head_dim: 192
qk_rope_head_dim: 64
qk_head_dim: 256
v_head_dim: 256
head_dim: 192                               # legacy field; effective per-head = 256
rope_interleave: True
partial_rotary_factor: 0.25
num_experts: 256
num_experts_per_tok: 8
n_shared_experts: 1
moe_intermediate_size: 2048
first_k_dense_replace: 3
topk_method: noaux_tc
scoring_func: sigmoid
norm_topk_prob: True
routed_scaling_factor: 2.5
n_group: 1
topk_group: 1
moe_layer_freq: 1
decoder_sparse_step: 1
mlp_only_layers: []
index_head_dim: 128
index_n_heads: 32
index_topk: 2048
index_share_for_mtp_iteration: False
n_routed_experts: 256
rope_parameters: {'rope_theta': 8000000.0, 'rope_scaling': None, 'rope_type': 'default'}
```

**No `expert_gating_func` field exists.**
