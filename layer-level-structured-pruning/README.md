# Layer-level structured pruning (ShortGPT on GLM-5.2)

Phase 7 → Phase 8 + post-mortem. 36 GB (15.4%) saved on the spaced-N=12 plan
while preserving all 9 smoke baselines + the 18.7k BLUE-FALCON retrieval
task. Failed contiguous-N=16 attempt documented as a forensic post-mortem
(see `traces/glm52-coding-en-cmp4386b_*.jsonl` for the 3-way comparison).

## What was pruned

Transformer-block (layer) level, not weights/neurons/experts. Selection
metric = ShortGPT Block Influence (BI) score = `1 - cos(l_in_residual,
l_out_residual)` per token per layer, averaged across 161 calibration prompts.

| Phase | Operation | Layers dropped | Tensors dropped | Size |
|---|---|---|---|---|
| Phase 7b | drop `blk.78.*` MTP layer | 1 (MTP) | 22 | 232 GB → 227 GB |
| Phase 8 | drop spaced-N=12 leftover layers | 12 normal | ~264 | 227 GB → 191 GB |
| **Total** | | **13 layers** | **~286 tensors** | **36 GB / 15.4% saved** |

Selection: `[3, 5, 11, 44, 51, 53, 56, 58, 60, 62, 64, 67]` — 12 lowest-BI
layers with no two adjacent (adjacency cap = 1, empirically validated safe).

## Scripts

| File | Purpose |
|---|---|
| `prune_gguf.py` | Generic GGUF tensor-pruning primitive (`prune_gguf()` with `tensor_name_remap` + `kv_overrides` hooks). Reads a `gguf.GGUFReader`, copies KV metadata + surviving tensors via `gguf.GGUFWriter`. Used standalone for the Phase 7b MTP prune. |
| `prune_layers.py` | Multi-shard layer-drop driver for GLM-5.2. Reads a BI plan JSON (produced by analyze_bi_scores.py), processes all 9 shards: drops `blk.N.*` for dropped layers, renumbers kept `blk.N.*` → `blk.{new_idx}.*`, patches `glm-dsa.block_count` + `split.tensors.count` (as INT32) in shard 1, renames `blk.78` MTP → `blk.{len(kept_normal)}`. Imports `prune_gguf` via `sys.path.insert(Path(__file__).parent)` — both scripts must stay co-located. |
| `analyze_bi_scores.py` | Aggregate per-layer BI scores from trace JSONL. CLI: `--top-n N` + `--max-contiguous-drops N` (default=1, empirically-validated safe pattern). Produces plan JSON with `layers_to_drop`, `renumber_map`, `block_count_new`. |

## Dry-run integration check

Run `prune_layers.py --dry-run` before writing pruned shards. It is idempotent:
it does not create output directories or GGUF shards. It eagerly imports
`prune_gguf`, validates the pruning hook API, loads the BI plan, scans source
GGUF shards, reports dropped/kept tensor counts, and previews `blk.78` MTP
renumbering. Exit code contract: `0` means all checks passed; any non-zero
status means dry-run failure or invalid CLI invocation and stderr names the
error.

## Reports (12)

- `glm52_shortgpt_bi_scores.{json,md}` — original top-N=16 contiguous plan
  (DROPPED 14 deep layers in a row; led to model collapse → documented the
  need for adjacency constraints).
- `glm52_shortgpt_bi_scores_spaced12.json` — the passing spaced-N=12 plan
  applied to the live pruned model.
- `glm52_shortgpt_bi_scores_v2.{json,md}` — post-mortem v2 plan with the
  `--max-contiguous-drops 1` default applied. The SAFE N=16 plan that would
  have prevented the Phase 8 first-attempt collapse if used initially.
- `glm52_shortgpt_bi_scores_topn{8,10}.*` — smaller-drop plans (ablation
  candidates; never applied).
- `glm52_prune_inventory.{json,md}` — Phase 7a full 1809-tensor inventory
  + categorized Tensor types (normal_blk / mtp / indexer / embed).
- `glm52_wanda_baseline.md` — pre-prune baseline performance reference
  (kept for future Wanda-on-pruned comparison).

## Traces (327 tracked files)

### `shortgpt_bi/calib/` (161 files, 377 MB)

161-prompt Block Influence calibration traces from the *unpruned* ShortGPT
source model. Each file = one prompt from the 7-language × 7-domain suite.
909,191 total records / 457,605 `block_influence` events. Wall time 6.2 min.

Used as the BI calibration input for `analyze_bi_scores.py`.

### `glm52-coding-en-cmp4386b_*.{jsonl,meta.json}` (9 files)

3-way forensic comparison against a single merge-sort prompt, captured with
`--trace-activation-stride 1` + `TRACE_MAX_TOKENS=0` (full per-token-per-layer
coverage) across 3 model variants:
- `cmp4386b_unpruned-*.jsonl` — unpruned ShortGPT source (227 GB)
- `cmp4386b_spaced12-*.jsonl` — passing spaced-N=12 pruned (191 GB)
- `cmp4386b_failcontig16-*.jsonl` — FAILED contiguous-16 pruned (191 GB,
  since-deleted but traces preserved as forensic evidence)

These 9 files are the only trace data from which the seam-mismatch root
cause (commit `ee3a6f5`) was diagnosed — `analyze_bi_scores.py`'s default
`--max-contiguous-drops 1` is derived directly from this comparison.

### `shortgpt_pruned_baselines/` (2 files)

PASS-model baseline outputs after the spaced-N=12 prune:
- `longctx_BLUE_FALCON_spacedN12.txt` — 18.7k-token BLUE-FALCON retrieval
  run, sentinel + function name + recursion flag all recovered
- `merge_sort_spacedN12.txt` — iterative merge-sort Python code, passes
  6/6 sanity tests

Re-run via:
```sh
MODEL=$MODEL_DIR/GLM-5.2-shortgpt-pruned-IQ2S-experts-IQ4NL-rest/GLM-5.2-shortgpt-00001-of-00009.gguf \
  bash ../common/baselines/glm52_merge_sort_baseline.sh
```

## Key learnings (recorded in GLM52_SESSION_MEMORY.md)

1. **ShortGPT's BI heuristic is per-layer correct but cannot pick contiguous
   layers naively.** BI of 0.025 looks redundant in isolation, but compounding
   residual-shift effects across consecutive layers create shockwaves. The
   `--max-contiguous-drops 1` default is baked into `analyze_bi_scores.py`
   based on this evidence.

2. **#4386 attention-sink is necessary but not sufficient indicator of model
   health.** Contiguous-14-layer drop preserves #4386 saturation at L36-L66
   (nearly identical to unpruned) but the model still collapses — the actual
   root cause is residual-stream seam-mismatch downstream.

3. **The accumulative cos change across 14 dropped layers in unpruned was
   ~30%.** Pruned model forces orig L50's residual directly into orig L65's
   input via New L40, making kept downstream layers (orig L65-L74) operate
   out-of-distribution for ~10 layers before re-converging.

See `../GLM52_SESSION_MEMORY.md` → "Phase 8" + "Phase 8 follow-up" sections
for the full narrative.
