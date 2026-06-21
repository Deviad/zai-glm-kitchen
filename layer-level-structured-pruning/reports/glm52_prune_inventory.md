# Phase 7a — GLM-5.2 prune inventory (non-destructive)

**Date:** 2026-06-20
**Scope:** Inventory pass only. No GGUF modification. Identifies which tensors
are **safe to prune from a baseline-inference GGUF** and which need
additional verification.

## TL;DR

| Category | Tensors | Total size | % of model | Loader flag | Forward-path use | Safe to prune? |
|---|---|---|---|---|---|---|
| Indexer `blk.0..77.indexer.*` | 390 | 413 MB | 0.17% | **REQUIRED** (flags=0) | Not invoked (Phase 3) | NO — would need loader patch |
| Indexer `blk.78.indexer.*` | 5 | 3 MB | 0.00% | NOT_REQUIRED | Not invoked | YES (negligible savings) |
| MTP `blk.78.*` (NextN experts+attn) | 22 | 5.60 GB | 2.25% | **NOT_REQUIRED + SKIP** | Not invoked (graph = deepseek2) | **YES — primary prune target** |
| Embed/output head | 3 | 1.32 GB | 0.53% | REQUIRED | Always used | NO |
| Normal `blk.0..blk.77` | 1389 | 241.85 GB | 97.06% | REQUIRED | Main model | NO |
| **TOTAL** | **1809** | **249.18 GB** | 100.00% | — | — | — |

**Recommended prune (MTP only, fully loader-safe):** drop all 22 `blk.78.*`
tensors → saves **5.60 GB (~2.25% of total)** with zero loader changes.

**MTP speedup is not attainable in current llama.cpp** — see "MTP draft
decode is not wired for GLM-DSA" below. Pruning has **zero opportunity cost**
as of 2026-06-20.

**Indexer prune OUT OF SCOPE:** documented here for completeness but
requires patching `llama-model.cpp` to flag `indexer_*` tensors as
`TENSOR_NOT_REQUIRED` for layers `i < n_layer`. The 417 MB savings
(0.17%) don't justify the loader-modification risk.

## Critical loader-code finding (refines the prune plan)

`/Users/spotted/projects/llama.cpp/src/models/glm-dsa.cpp` `load_arch_tensors`:

```cpp
for (int i = 0; i < n_layer_all; ++i) {
    int flags = 0;
    if (i >= n_layer) {
        // skip all tensors in the NextN layers
        // TODO @ngxson : TENSOR_NOT_REQUIRED was a hack, need to remove it later
        flags |= TENSOR_SKIP | TENSOR_NOT_REQUIRED;
    }
    // ... all block tensors for layer i use `flags`
    // ... including indexer.* and blk.* and nextn.*
}
```

Two implications:

1. **All blk.78.* tensors (including its indexer subset) are flagged
   `TENSOR_SKIP | TENSOR_NOT_REQUIRED`** — the loader tolerates their
   absence. The maintainer comment "preserved but unused" is explicit.
   Pruning blk.78.* from the GGUF is loader-safe.

2. **Indexer tensors in blk.0..blk.77 are flagged REQUIRED (flags=0)** —
   the loader asserts their presence at load time. They're loaded into
   memory but, per Phase 3 empirical evidence, never invoked during the
   default forward path (`deepseek2::graph is_lite=false` MLA path).
   Pruning them would require a loader patch marking them `TENSOR_NOT_REQUIRED`
   for layers `i < n_layer`. Risk + complexity not worth 417 MB.

## Method

1. Sweep all 9 shards of
   `/Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/`
   via `gguf-py` `GGUFReader`.
2. For each tensor record: name, shard, shape, quant type, byte count
   (computed from `GGML_QUANT_SIZES` block-size + type-size table).
3. Categorize by name pattern. Cross-reference Phase 3 empirical evidence
   (`phase3_dsa_unblock/README.md`) for "loaded but never used during
   normal inference" claim.
4. NO new C++ runs. NO model load. NO actual pruning. Pure metadata scan.

## Per-shard category breakdown (GB)

```
shard                                              normal    mtp   index  embed    TOTAL
GLM-5.2-mixed-00001-of-00009.gguf                    0.00   0.00   0.000   0.00     0.00   (metadata-only shard)
GLM-5.2-mixed-00002-of-00009.gguf                   30.73   0.00   0.063   1.32    32.11   (token_embd lives here)
GLM-5.2-mixed-00003-of-00009.gguf                   33.20   0.00   0.053   0.00    33.26
GLM-5.2-mixed-00004-of-00009.gguf                   33.29   0.00   0.058   0.00    33.35
GLM-5.2-mixed-00005-of-00009.gguf                   33.20   0.00   0.053   0.00    33.25
GLM-5.2-mixed-00006-of-00009.gguf                   33.20   0.00   0.053   0.00    33.26
GLM-5.2-mixed-00007-of-00009.gguf                   33.29   0.00   0.058   0.00    33.35
GLM-5.2-mixed-00008-of-00009.gguf                   33.20   0.00   0.053   0.00    33.25
GLM-5.2-mixed-00009-of-00009.gguf                   11.73   5.60   0.026   0.00    17.36   (MTP layer lives here)
TOTAL                                              241.85   5.60   0.417   1.32   249.18
```

Indexer weights are spread across shards 2-9 (5 subcomponents × 79 layers each
≈ 395 tensors, ~50 MB per shard). MTP is **entirely in shard 9**.

## Category 1: Indexer (DSA lightning-indexer weights) — DEPRIORITIZED

**Empirical evidence (Phase 3) — forward-path unused:**

- Phase 3 patch (`phase3_dsa_unblock/`) activates the DSA indexer forward path
  by aliasing `glm_dsa::graph` from `deepseek2::graph` to `deepseek32::graph`
  and extending the Hadamard rotation gate to fire for `LLM_ARCH_GLM_DSA`.
- With the patch active, the merge-sort baseline drops from 20.4 → 8.3 tok/s
  (**-59% generation speed**) — proves the extra `mul_mat + Hadamard +
  ggml_top_k` per decoder layer runs when the indexer is enabled.
- Without the patch (default / current baseline), long-ctx retrieval baseline
  (18,745 prompt tokens) works correctly at 11.4 tok/s gen and recovers the
  BLUE-FALCON-48217 sentinel. **The indexer weights are not invoked.**

**But the loader still requires their presence for blk.0..blk.77** — the model
loader marks the `indexer_*` tensors REQUIRED for normal layers (`flags=0`).
Pruning them at the GGUF level without patching the loader would fail at
`create_tensor(...)` with `tensor not found in file`.

**Decision:** out of scope for Phase 7b. Documented here for a future Phase 7c
("indexer-tolerant loader") if 417 MB ever becomes worth the risk.

### Indexer subcomponent breakdown

```
suffix                              count       MB       notes
indexer.attn_q_b.weight                79   372.77       89.5% of indexer category
indexer.attn_k.weight                  79    34.95
indexer.proj.weight                    79     0.11       (8.74 MB total)
indexer.k_norm.bias                    79     0.04
indexer.k_norm.weight                  79     0.04
TOTAL                                 395   416.53
```

5 uniform subcomponents × 79 layers (blk.0..blk.78). The 5 tensors that live in
blk.78 are already covered by the MTP prune below (same loader flag).


## Category 2: MTP (blk.78 — Multi-Token-Prediction layer) — **PRIMARY PRUNE TARGET**

**Loader status (verified in `glm-dsa.cpp` `load_arch_tensors`):**

All 22 `blk.78.*` tensors are flagged with `TENSOR_SKIP | TENSOR_NOT_REQUIRED`.
The loader tolerates their absence; comment in source reads:
> "NextN/MTP tensors (preserved but unused) - conditionally load for last n_layer_nextn"

**Forward-path status (from Phase 3 / graph alias):**

Phase 3 revert left `glm_dsa::graph` aliased back to `deepseek2::graph`. The
deepseek2 graph invokes layers `0..n_layer-1` (i.e., blk.0..blk.77) in the
standard forward pass; `blk.78` is only invoked through `embeddings_nextn_masked`
(sourced from `cparams`) which is off by default. So blk.78 is loaded but
not invoked for standard greedy/sampling inference.

**Decision:** primary prune target. Drop all 22 `blk.78.*` tensors. Saves
5.60 GB (2.25% of total) with zero loader modifications.

### MTP draft decode is NOT wired for GLM-DSA (verified 2026-06-20)

Two independent confirmations that the 22 MTP tensors cannot currently
produce any spec-decode speedup:

**1. Code-level — no `graph_mtp` for GLM-DSA:**

```cpp
// src/models/glm-dsa.cpp
std::unique_ptr<llm_graph_context> llama_model_glm_dsa::build_arch_graph(
    const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);  // always deepseek2::graph
}
```

GLM-DSA ignores `params.gtype`. Only 4 architectures implement
`LLM_GRAPH_TYPE_DECODER_MTP`:
- `cohere2moe::graph_mtp`
- `qwen35::graph_mtp`
- `qwen35moe::graph_mtp`
- `step35::graph_mtp`

GLM family (glm-dsa, glm4, glm4-moe, chatglm) has zero `graph_mtp`
implementations. So `--spec-type draft-mtp` cannot build an MTP draft
graph from GLM-5.2's blk.78 weights.

**2. Empirical — `--spec-type draft-mtp` is silently ignored + makes gen SLOWER:**

Same GLM-5.2 baseline, identical prompt `"Write one word: hi"`, -n 5:

| Mode | Prompt t/s | Gen t/s |
|---|---|---|
| `--spec-type none` (baseline) | 39.2 | **25.6** |
| `--spec-type draft-mtp --spec-draft-n-max 3` | 38.0 | **6.9** (4× slower) |
| `--spec-type draft-mtp --spec-draft-n-max 1` | 34.5 | **7.2** (3.5× slower) |

The 4× slowdown is the spec-decode round-trip overhead incurred without
any draft acceptance benefit. The blk.78 weights never fire to produce
a useful draft token.

**Conclusion:** pruning blk.78.* has **zero opportunity cost** as of this
build. Only restoration cost would be re-adding the tensors IF upstream
adds a GLM-DSA `graph_mtp` — mitigate by keeping the original shard 9
unmodified on disk.

### MTP subcomponent breakdown

```
suffix                                       count        MB
attn_k_b.weight                                  1       3.54
attn_kv_a_mqa.weight                             1       1.99
attn_kv_a_norm.weight                            1       0.00
attn_norm.weight                                 1       0.02
attn_output.weight                               1      56.62
attn_q_a.weight                                  1       7.08
attn_q_a_norm.weight                             1       0.01
attn_q_b.weight                                  1      18.87
attn_v_b.weight                                  1       4.72
exp_probs_b.bias                                 1       0.00
ffn_down_exps.weight                             1    1811.94   <- 1.81 GB expert tensor
ffn_down_shexp.weight                            1       7.08
ffn_gate_exps.weight                              1    1811.94   <- 1.81 GB expert tensor
ffn_gate_inp.weight                              1       6.29
ffn_gate_shexp.weight                             1       7.08
ffn_norm.weight                                   1       0.02
ffn_up_exps.weight                                1    1811.94   <- 1.81 GB expert tensor
ffn_up_shexp.weight                               1       7.08
nextn.eh_proj.weight                              1      42.47
nextn.enorm.weight                                1       0.02
nextn.hnorm.weight                                1       0.02
nextn.shared_head_norm.weight                     1       0.02
TOTAL                                           22    5598.77
```

**The 3 expert tensors account for 5.44 GB of the 5.60 GB MTP total (97%).**
Pruning just those 3 tensors = 97% of MTP savings while leaving the small
attention + nextn.norm tensors in place is the lowest-risk version.

## Category 3: Embed / output head — DO NOT PRUNE

| Tensor | Bytes | Purpose |
|---|---|---|
| `token_embd.weight` | 0.54 GB | Input embedding lookup |
| `output.weight` | 0.78 GB | LM head |
| `output_norm.weight` | ~24 KB | Final norm |

Always used; pruning is fatal.

## Category 4: Normal layers (`blk.0..blk.77`) — DO NOT PRUNE

1389 tensors, 241.85 GB. The actual model. Includes:
- Attention Q/K/V (absorbed MLA variants: `attn_q_a`, `attn_q_a_norm`,
  `attn_q_b`, `attn_k_b`, `attn_v_b`, `attn_kv_a_mqa`, `attn_kv_a_norm`)
- Attention output (`attn_output`, `attn_norm`)
- MoE FFN experts (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`)
- Shared experts (`ffn_*_shexp`)
- Per-layer norms and router (`ffn_gate_inp`, `ffn_norm`, `exp_probs_b.bias`)

Pruning any of these destroys model quality.

## Recommended Phase 7b plan (refined: MTP-only)

**Step 1 — Write prune tool** (~100 LoC, ~20 min):
- `scripts/prune_gguf.py`: takes an input GGUF path and a tensor-drop pattern
  (e.g., `blk.78.*`), outputs a new GGUF that omits the matching tensors.
- Use `gguf-py` `GGUFReader` to enumerate tensors + KV pairs; build a new
  GGUF via `GGUFWriter` mirroring the existing metadata, then copy tensors
  that don't match the drop pattern.
- Must preserve all KV pairs (`glm-dsa.expert_count`, `.nextn_predict_layers`,
  etc.) — only tensor sections are dropped. The loader reads these KV pairs
  for buffer sizing but doesn't require the `blk.78.*` tensors themselves.
- Verify shape: drop 22 `blk.78.*` tensors → output shard 9 should shrink
  by 5.6 GB (17.36 GB → 11.76 GB).

**Step 2 — Prune shard 9 + verify load** (~10 min):
- Run prune tool on `GLM-5.2-mixed-00009-of-00009.gguf` (the MTP-only shard).
- Try loading the pruned GGUF via `llama-cli --model <pruned-shard-9-and-unmodified-others>`.
- Expected: loader prints `n_layer_all = 79`, then for `i >= n_layer`, marks
  all `blk.78.*` tensors NOT_FOUND but tolerates them (no assert).
- If loader fails: refine pattern or check KV metadata for nextn_predict_layers.

**Step 3 — Baseline equivalence test** (~20 min, ~2 min wall):
- Run `scripts/baselines/glm52_merge_sort_baseline.sh` against the pruned model.
- Run `scripts/baselines/glm52_longctx_retrieval_baseline.sh` against the pruned model.
- Compare output text + perf numbers to the known-good baselines
  (`~31.5 t/s prompt, ~20.2 t/s gen` for merge-sort; `77 t/s / 11.4 t/s`
  + sentinel BLUE-FALCON-48217 recovery for long-ctx).
- Pass criteria: byte-identical output text (minus timing noise) + perf within
  ±5% of baseline. If pass: commit the pruned model as `GLM-5.2-pruned-mixed-IQ2S-...`.

**Step 4 — (deprecated) Loader-tolerant indexer prune**: skipped per TL;DR.

## Risks

1. **gguf-py writer tensor-by-tensor copy is slow for multi-GB files.**
   Need to use chunked writes or memory-mapped I/O. For 5.6 GB it's tractable
   (~1 min on NVMe).

2. **`glm-dsa.nextn_predict_layers` KV (if set to 1)** tells the loader to
   expect blk.78 tensors. We keep the KV but omit the tensors — the loader
   should tolerate this because `TENSOR_NOT_REQUIRED | TENSOR_SKIP`. If not,
   drop the KV too (or patch it to 0).

3. **GGUF shard metadata may need `tensor_count` updated** — `gguf-py`
   writer should handle this automatically when copying tensors. Verify by
   reading the output shard's `GGUF.tensor_count` field.

4. **Sharding alignment** — shard 9 is the only shard containing MTP. After
   pruning, its tensor count drops from 107 to 107-22 = 85. Other shards
   are unmodified. Verify the multi-shard loader handles heterogeneous
   `tensor_count` per shard. (It should — the loader concatenates tensors
   across shards by name, not by shard-local count.)

## Provenance

- Full tensor inventory: `reports/glm52_prune_inventory.json` (machine-readable, 1809 tensors)
- Phase 3 DSA evidence: `phase3_dsa_unblock/` (5 baseline scripts + README)
- AGENTS.md quantization policy: `blk.78 MTP routed experts: IQ4_NL exception; normal routed expert MLPs: IQ2_S`
- Tracing infrastructure: `src/gguf2mlx/tracing/` (126 tests pass)
- All baselines verify the known-good model output and timing.
