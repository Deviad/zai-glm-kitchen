# DSA / IndexShare / CSA research library

Source PDFs for the GLM-5.2 forward-path implementation work (Story 7.M /
IndexShare blocker in `REAP37_EXPERIMENTS.md` / `PLAN.md` §8 item 2).

All papers were fetched on 2026-06-21 directly from arXiv. Each PDF below is the
canonical arxiv PDF; titles verified by re-fetching the abstract page. Do not
rewrite or paraphrase the abstracts below — they are direct quotes from each
paper's page 1.

## Why this folder exists

Stock `mlx_lm` (`vendor/mlx-lm/mlx_lm/models/glm_moe_dsa.py` →
`deepseek_v32.Model`) and stock `llama.cpp` (`vendor/llama.cpp` `glm-dsa.cpp:152`
aliased to `deepseek32::graph`) **both run GLM-5.2 as plain MLA** — the DSA
lightning indexer tensors are loaded into memory but never executed in the
forward pass. The shortgpt-pruned GGUF carries 330 DSA indexer tensors across
66 blocks; if a future task is "make GLM-5.2 actually do DSA correctly," these
six papers are the mathematical ground truth for that work.

Cross-reference: `PLAN.md` §10 (research sources) and §7.M (mixed-precision MLX
export story, AC7 = IndexShare load caveat), `REAP37_EXPERIMENTS.md` (the
documented failure of the compat-hack workaround at long context).

## Papers (file | citation | abstract quote | what to extract)

### 1. `papers/deepseek-v3.2-dsa-2512.02556.pdf`

**Citation.** DeepSeek-AI. *DeepSeek-V3.2: Pushing the Frontier of Open Large
Language Models.* arXiv:2512.02556v1 [cs.CL], 2 Dec 2025.

**Abstract quote.** "(1) DeepSeek Sparse Attention (DSA): We introduce DSA, an
efficient attention mechanism that substantially reduces computational
complexity while preserving model performance in long-context scenarios."

**What to extract.** This is the **DSA origin paper** — the lightning-indexer
equation, indexer-head count H_I, the top-k selection, and how DSA sparsifies
the MLA path. Read this first. Required to derive what
`attn_q_b`, `attn_k`, `k_norm.{weight,bias}`, and `proj` actually *compute* in
the per-layer indexer block.

### 2. `papers/glm5-tech-report-2602.15763.pdf`

**Citation.** GLM-5 Team (Zhipu AI & Tsinghua University). *GLM-5: from Vibe
Coding to Agentic Engineering.* arXiv:2602.15763v2 [cs.LG], 24 Feb 2026.

**Abstract quote.** "Building upon the agentic, reasoning, and coding (ARC)
capabilities of its predecessor, GLM-5 adopts DSA to significantly reduce
training and inference costs while maintaining long-context fidelity."

**What to extract.** Canonical architectural description of GLM-5 / 5.2:
MLA + DSA + MoE + MTP fusion. Use to verify `config.json` shape defaults from
the converter (PLAN.md §7, AC2) and to confirm the MTP `nextn_predict_layers`
KV-share convention.

### 3. `papers/indexcache-indexshare-2603.12201.pdf`

**Citation.** Bai, Dong, Jiang, Lv, Du, Zeng, Tang, Li. *IndexCache:
Accelerating Sparse Attention via Cross-Layer Index Reuse.* arXiv:2603.12201v1
[cs.CL], 12 Mar 2026. Tsinghua University + Z.ai.

**Abstract quote.** "We present IndexCache, which exploits this cross-layer
redundancy by partitioning layers into a small set of Full layers that run
their own indexers and a majority of Shared layers that simply reuse the
nearest Full layer's top-k indices."

**What to extract.** This is **the IndexShare paper** — defines the **F/S
layer pattern** that GLM-5.2 ships with. Direct quote confirms the
mechanism: Full (F) layers own indexer tensors; Shared (S) layers **lack
indexer tensors entirely** and reuse the nearest preceding F layer's top-k
selection. Required for resolving the AGENTS.md "anomaly" (whether the 330
indexer tensors found by the tooling scan are F-only or materialized on all
layers). Also benchmarks 75% indexer removal at negligible quality loss on a
30B DSA model — i.e. the IndexShare pattern is well-verified, not a hack.

### 4. `papers/streamindex-v4-csa-2605.02568.pdf`

**Citation.** Jaber & Jaber (RightNow AI). *StreamIndex: Memory-Bounded
Compressed Sparse Attention via Streaming Top-k.* arXiv:2605.02568v1 [cs.LG],
4 May 2026.

**Abstract quote.** "DeepSeek-V3.2 and V4 introduce Compressed Sparse Attention
(CSA): a lightning indexer (a learned scoring projection over compressed keys)
scores them, the top-k are selected per query, and a sparse attention kernel
reads only those. … S TREAM I NDEX, a Triton implementation of the CSA pipeline
whose central component is a chunked partition-merge top-k driver that never
materializes the full intermediate."

**What to extract.** CSA (V4 successor to DSA) — the indexer scoring equation
`I(t,s) = Σ_h w_{t,h} · ReLU(q_{t,h} · K_s)`, the per-head weighting, and how
top-k is selected. Implementing GLM-5.2's indexer forward in MLX or llama.cpp
must reproduce this scoring. Has a working Triton reference impl to diff
against.

### 5. `papers/flashmemory-deepseek-v4-2606.09079.pdf`

**Citation.** *FlashMemory-DeepSeek-V4: Lightning Index Ultra-Long Context via
Lookahead Sparse Attention.* arXiv:2606.09079v1, Jun 2026.

**What to extract.** V4 hybrid HCA + CSA architecture ("128:1 HCA compression"
per PLAN.md §10). Relevant mostly as forward-looking reference — GLM-5.2 itself
is V3.2-class DSA, not V4 CSA, so this is the *next-generation* picture of
where sparse-attention is heading, not the GLM-5.2 spec directly.

### 6. `papers/misa-dsa-repro-2605.07363.pdf`

**Citation.** Zhou, Meng, Xu, Liu, Lu, Zhang, Pei. *MISA: Mixture of Indexer
Sparse Attention for Long-Context LLM Inference.* arXiv:2605.07363v1 [cs.LG],
8 May 2026. https://github.com/MuLabPKU/TransArch

**Abstract quote.** "DeepSeek Sparse Attention (DSA) sets the state of the art
for fine-grained inference-time sparse attention by introducing a learned
token-wise indexer that scores every prefix token and selects the top-k for the
main attention."

**What to extract.** Independent third-party DSA indexer re-implementation,
with a TileLang kernel that "delivers roughly a 3.82× speedup over DSA's
original indexer kernel on a single NVIDIA H200 GPU." Per the LOE estimate
in the prior plan, this is the closest published ground-truth reference for
the DSA forward path; use as cross-check against DeepSeek-V3.2's primary
description. Code link above may supply working indexer kernels.

## Recommended reading order for the forward-path work

1. `indexcache-indexshare-2603.12201.pdf` — defines the F/S layer pattern.
   Read first to resolve the "330 indexers on 66 layers" anomaly (AC7 in
   PLAN.md §7.M).
2. `deepseek-v3.2-dsa-2512.02556.pdf` — the DSA lightning-indexer math.
3. `misa-dsa-repro-2605.07363.pdf` — independent re-interpretation + workable
   kernel; closest thing to a reference impl.
4. `glm5-tech-report-2602.15763.pdf` — verify GLM-5.2's exact MTP/MoE
   conventions against Zai's own description.
5. `streamindex-v4-csa-2605.02568.pdf` + `flashmemory-deepseek-v4-2606.09079.pdf`
   — next-gen context (CSA / V4), not strictly needed for GLM-5.2 (V3.2-class)
   but useful for scope-checking against DeepSeek-V4 directions.
