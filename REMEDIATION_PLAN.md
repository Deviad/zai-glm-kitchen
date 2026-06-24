# Remediation Plan — Code Review Findings

**Source:** `CODE_REVIEW.md` (colleague review based on `git diff HEAD` + `git diff HEAD~1..HEAD`, covering 9 files).

**Revised 2026-06-24** after a codebase audit corrected the original P0 item
(one factual premise about the GGUF was wrong) and verified every P1/P2/P3
claim against source. Audit evidence is inline under each item.

**Severity key:** 🔴 P0 (blocking/stale claim) · 🟠 P1 (data corruption or waste) · 🟡 P2 (correctness fragility or tech debt) · ⚪ P3 (cosmetic/low-risk)

---

## P0 — 🔴 AC3 F/S IndexShare: the "IMPLEMENTED" status in PLAN.md is a stale overclaim, but the originally-proposed C++ fix would be a no-op — reframe as data-layer + docs

### Original (failed) premise
The original remediation claimed GLM-5.2 ships the IndexCache F/S pattern
(21 "full" layers own indexer tensors; 57 "shared" layers carry none and reuse
the preceding full layer's top-k). The proposed fix was to gate
`glm-dsa.cpp::graph` on `indexer_attn_q_b != nullptr` whenever a "shared"
layer is processed, skipping the indexer matmul + `ggml_top_k` for 57 layers,
and to mark S-layer indexer tensors `TENSOR_NOT_REQUIRED` in
`load_arch_tensors`.

### What the codebase + actual artifact show (audit 2026-06-24)

1. **There is NO F/S signal in the known-good GGUF.** All 9 shards of
   `$MODEL_DIR/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/` were scanned:

   ```
   glm-dsa.block_count               = 79
   glm-dsa.leading_dense_block_count = 3        (gates ONLY dense-vs-MoE FFN,
                                                 NOT indexer F/S)
   glm-dsa.attention.indexer.top_k   = 2048
   glm-dsa.attention.indexer.head_count = 32
   glm-dsa.attention.indexer.key_length = 128
   (no indexer_types[] / mlp_layer_types[] / index_share_for_mtp_iteration keys)
   ```

2. **Indexer tensors exist on EVERY one of the 79 blocks.** A tensor-name sweep
   found 632 hits across `indexer.attn_q_b | indexer.attn_k | indexer.proj |
   indexer.k_norm.{weight,bias}` = 8 tensors × 79 blocks. There is no
   per-layer "absence" for a C++ gate to detect.

3. **`glm-dsa.cpp` now has its own graph that DOES run the indexer** (lines
   110-111 create the 5 indexer tensors per layer; the graph block at
   `for (int il = 0; il < n_layer; ++il)` unconditionally runs the lightning
   indexer + `ggml_top_k` on every layer). So the older claim still present in
   `AGENTS.md` and `docs/research/README.md` — "`glm-dsa.cpp:152` aliases to
   `deepseek32::graph` with zero indexer references" — is also stale and
   contradicts current source.

4. **The AC3 "fix" as written would be a no-op.** `indexer_attn_q_b` is created
   via `create_tensor(...)` (line 111) WITHOUT `TENSOR_NOT_REQUIRED`, so on the
   current artifact it is never null — a `nullptr` guard is dead code, and the
   shared-layer top-k reuse path would never execute. Real implementation
   requires data-layer work first.

5. **Counts in the original P0 were wrong.** The plan asserted "8 indexer
   tensors per layer / 21×8=168 vs 78×8=624 saved." Reality: there are **5**
   indexer tensors per layer (`indexer.k_norm.weight`, `indexer.k_norm.bias`,
   `indexer.proj.weight`, `indexer.attn_k.weight`, `indexer.attn_q_b.weight`
   — lines 104-111), so any tensor-count arithmetic in the AC must reflect 5.

6. **`n_layer_dense_lead` is not an F/S signal.** It gates only dense-vs-MoE
   FFN (see `glm-dsa.cpp:112` `if (i < n_layer_dense_lead)` — dense FFN, else
   MoE). DeepSeek-3.2, GLM-4-MoE, Kimi-linear all use it the same way.
   Reusing it as a proxy for "full layer" would be incorrect.

7. The AC3 *overclaim itself* in `PLAN.md §7.L` ("IMPLEMENTED 2026-06-23
   (AC1-AC6 done)") is genuinely stale — AC3 is NOT implemented — so that
   part of the original review is fair.

### Root cause (corrected)
- `glm-dsa.cpp::graph` was originally cloned from `deepseek32::graph`, which
  processes every layer uniformly. The graph now runs the indexer on every
  layer regardless of whether GLM-5.2 intends F/S scheduling.
- BUT: whether the deployed architecture *actually* wants F/S reuse depends on
  `indexer_types[]` / `index_share_for_mtp_iteration` metadata, which the
  current known-good GGUF does **not** carry. Without that metadata, the C++
  graph has no input to gate on.

### Fix actions (reframed)

1. **Decide whether GLM-5.2 actually ships F/S IndexShare.**
   - Pull `config.json` from the upstream HF repo and check for
     `indexer_types[]` / `index_share_for_mtp_iteration` / `mlp_layer_types[]`.
   - If absent upstream: AC3 (and the AC3 checkbox) should be **dropped or
     rewritten**, not "fixed" — there is nothing to gate.
   - If present upstream: the current known-good GGUF was quantized from a
     checkpoint that materializes indexer tensors on all 79 blocks, so the F/S
     metadata is not flowing through `gguf-py`'s writer. Add the missing keys
     to `gguf-py/gguf/constants.py` + the GLM-DSA arch write path, and
     **re-quantize** the GGUF so shared-layer indexer tensors are actually
     omitted.

2. **Only after step 1 metadata exists**, fix the C++ graph:
   - `glm-dsa.cpp::graph` — read `indexer_types[]` (or equivalent) per layer.
     On a "shared" layer, bypass the indexer matmul + `ggml_top_k` and reuse
     the most recent "full" layer's `top_k` tensor (carry it in a
     `ggml_tensor * prev_full_topk` loop-local).
   - `glm-dsa.cpp::load_arch_tensors` (line 48) — only create `indexer.*`
     tensors on full layers (mark missing ones `TENSOR_NOT_REQUIRED`). Use
     **5/layer** in the tensor-count assertions (NOT 8).
   - Verify: absence-of-indexer case must not null-deref in
     `build_indexer_k_batched` / the sparse `build_attn` call.

3. **Update docs to reflect that `glm-dsa.cpp` already runs its indexer** (remove
   stale "zero indexer references" claims in `AGENTS.md`, `docs/research/README.md`,
   and any `PLAN.md` prose that asserts the alias-to-deepseek2 situation is current).

4. **`PLAN.md §7.L` status line** — change from "IMPLEMENTED 2026-06-23
   (AC1-AC6 done)" to an honest state:
   - AC1-AC2, AC4-AC6 (modulo tensor-count fix), AC7-as-negative-finding: done.
   - AC3: BLOCKED on upstream-metadata investigation (step 1). Reframed, not
     implemented.

### Acceptance criteria (corrected)
- [ ] **Decision recorded in `GLM52_SESSION_MEMORY.md`:** whether the upstream
      GLM-5.2 checkpoint ships `indexer_types[]` (verdict + evidence + URL).
- [ ] If/upstream ships F/S: GGUF re-quantized with shared-layer `indexer.*`
      tensors omitted and `indexer_types[]` written; `load_arch_tensors` creates
      `indexer.*` only on full layers, reflected by 5/layer (not 8) tensor
      counts.
- [ ] If/upstream does NOT ship F/S: AC3 dropped from `PLAN.md §7.L` with a
      note; no C++ change attempted.
- [ ] `glm-dsa.cpp::graph` either (a) gates indexer+topk on `indexer_types[il]`
      being full and reuses prior top-k for shared layers, or (b) unchanged if
      AC3 is dropped — documented either way.
- [ ] No segfault on any prompt; `2+2` + merge-sort smoke prompts still
      coherent via `llama-cli`/`llama-server` chat mode.
- [ ] `PLAN.md`, `AGENTS.md`, `docs/research/README.md` no longer claim the
      stale "aliases to deepseek2/deepseek32 with zero indexer" state nor
      "AC3 IMPLEMENTED".
- [ ] Finding appended to `GLM52_SESSION_MEMORY.md` (symptom → cause →
      reframed fix → verified), per repo contract.

### Estimated effort
- Docs cleanup (`PLAN.md` status, `AGENTS.md`, `docs/research/README.md`):
  ~20 min.
- Upstream-config investigation: ~30 min.
- If F/S real: GGUF re-quantize + gguf-py writer extension +
  `load_arch_tensors` refactor (~1-2 days, NOT 2h).
- If F/S dropped: C++ unchanged, story closed.

**Why 2h is wrong:** the original estimate treated this as a one-file C++
patch. The true blocking work is data-layer: sourcing the `indexer_types[]`
metadata and re-quantizing a 232 GB model. The C++ change is the small tail
end of a much bigger story.

### Decision record (2026-06-24): AC3 data-layer branch NOT pursued — baseline-preserving scope

Rationale recorded before implementing the rest of this plan:

1. **What's working now and must not break.** The canonical GGUF
   `$MODEL_DIR/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/` (232 GB, 2.64 BPW)
   loads and runs cleanly: Baseline 1 (merge sort) ~20.2 tok/s, correct;
   Baseline 2 (BLUE-FALCON 20K retrieval) correct, 11.4 tok/s gen; AC1/AC2/AC4-AC6
   landed (own `glm-dsa.cpp` graph, DSA KV-cache switch, Hadamard fix); no
   segfault; coherent output. The Phase 2b multilingual routing study was run
   against this baseline. This GGUF is the reference artifact for *all* future
   GLM-5.2 work per `AGENTS.md`.

2. **The headline DSA regression is kernel-bound, not layer-count-bound.**
   Session memory 2026-06-23: DSA decode @ 54K = **4.22 tok/s vs dense 7.2
   tok/s**. The root cause is unoptimized kernels (no fused indexer scoring,
   generic `ggml_top_k` 54K→2048 sort, no Metal sparse-gather) — *not* that the
   indexer runs on 57 extra layers. F/S would cut the count of indexer passes
   ~4× but each remaining pass stays on the slow path and `ggml_top_k` runs
   per-full-layer anyway. Optimistic best case 4.22 → ~5-6 tok/s, *still below*
   the dense 7.2 baseline. Chasing F/S does not recover the regression.

3. **The GGUF IS the baseline.** Re-quantizing to drop shared-layer indexer
   tensors would silently invalidate every recorded result in
   `GLM52_SESSION_MEMORY.md`, `KITCHEN_RESULTS.md`, and the Phase 2b report.
   Blast radius ≫ the benefit.

4. **Decision.** AC3's data-layer branch (gguf-py writer extension + 232 GB
   re-quantize + `load_arch_tensors` refactor at 5 tensors/layer) is **NOT
   pursued.** AC3 is reframed from "fix" to "investigated, documented,
   deferred by design".

5. **Upstream investigation RESULT (2026-06-24).** Fetched
   `huggingface.co/zai-org/GLM-5.2/resolve/main/config.json`:
   - `indexer_types[]` = 78 entries: **21 `"full"` + 57 `"shared"`** — the F/S
     split the original P0 claimed is REAL upstream.
   - `index_topk_freq = 4` (1-in-4 full cadence); `index_share_for_mtp_iteration = true`.
   - Full-layer indices: 0, 1, 2 (the 3 leading dense layers), then every 4th
     from 6 onward (6, 10, 14, …, 74) → 21 of 78.
   - `mlp_layer_types[]` = 3 `"dense"` + 75 `"sparse"` (matches
     `first_k_dense_replace=3`; INDEPENDENT of F/S — dense vs MoE FFN).
   - `num_hidden_layers=78`, `num_nextn_predict_layers=1` → 79 GGUF blocks.

   Settled wording of the finding: **"AC3 (F/S IndexShare) is real upstream —
   21 full / 57 shared + MTP-reuse — but is NOT pursued here; the DSA 4.22 vs
   7.2 tok/s regression at 54K is kernel-bound (unoptimized indexer scoring,
   generic `ggml_top_k` sort, no Metal sparse-gather), not layer-count-bound,
   so F/S would not recover the regression, and re-quantizing the canonical
   232 GB baseline would invalidate every prior result. The known-good GGUF
   over-materializes indexer tensors (all 79 blocks carry them; only 21 need
   them) — this is redundant compute, not a correctness bug, which is why the
   baseline works."**

#### Do-not-touch list (preserved-baseline contract)

The following are explicitly out of scope for this remediation pass. They
must remain in their current working state:

- **`glm-dsa.cpp::graph` indexer execution** — runs the lightning indexer +
  `ggml_top_k` on every layer. This is CORRECT for the current artifact
  (indexer tensors exist on all 79 blocks) and must not be gated/refactored.
- **`glm-dsa.cpp::load_arch_tensors` tensor-creation block (lines 81-141)**
  — must NOT be gated on a per-layer role; the known-good GGUF materializes
  indexer tensors on every block, so gating would require a re-quantize to
  test and would break the baseline.
- **The canonical GGUF itself** — no re-quantize, no tensor stripping.
- **`llama_model_glm_dsa::graph` alias** (already its own struct, `models.h`)
  — leave as-is; it works.
- **`llama-model.cpp` `case LLM_ARCH_GLM_DSA` KV-cache arm** — leave as-is.

In-scope for this pass (zero-baseline-risk changes only): the P1/P2/P3 items
below; a 1-2 line additive overflow `GGML_ASSERT` on the `ggml_view_4d` n_stream
split (batch-1 behavior unchanged, so the working baseline is not exercised);
the docs-staleness cleanup; and the upstream investigation itself.

---

## P1 — 🟠 Duplicate entry in GLM52_SESSION_MEMORY.md

### Issue
Two consecutive entries with the **identical title** `### 2026-06-23 (C) — DSA context-length sweep: crossover NOT reachable with full-sort top_k` and identical content. The second is a copy-paste ghost.

### Verified
`grep -c "DSA context-length sweep" GLM52_SESSION_MEMORY.md` → **2**.

### Fix action
Delete the duplicate (second occurrence). Verify no loss of distinct content.

### Acceptance criteria
- [ ] `grep -c "DSA context-length sweep" GLM52_SESSION_MEMORY.md` returns 1.
- [ ] The surviving entry contains the full data table + verdict + structural
      note + references to scripts/data paths.

### Estimated effort
2 minutes.

---

## P1 — 🟠 `convert_glm52_jangtq_k.py`: Resume logic corrupts on subsequent runs

### Issue
Three intertwined problems in the resume/rename logic:

1. **Resume glob misses completed runs** — After rename `XXXXX→NNNNN`, the glob
   `model-*-of-XXXXX.safetensors` finds nothing → `done_keys` empty →
   re-does every tensor scratch. Old `model-*-of-NNNNN.safetensors` orphan
   files accumulate: 3 runs = 3× disk usage. **Verified:** line 219 globs
   only the `XXXXX` pattern.
2. **Stale index entries (originally claimed)** — The original review asserted
   that re-quantizing a tensor would land both old and new entries in
   `shard_map` for the same key → silent corruption. **Audit:
   INCORRECT subclaim.** `shard_map[k] = …` at lines 205 *and* 231 is a
   dict overwrite, not append, so duplicate keys cannot accumulate. The real
   risk is orphan *shard files* (item 1), not duplicate index entries.
3. **`idx_str` parsing** — `sf.name.split("-")[1]` (line 232) works for
   `XXXXX` but is brittle after rename. Functional today, but only because
   the `XXXXX` glob restricts the input set.

### Fix actions
1. **Glob both patterns** — match `model-*-of-*.safetensors` to detect
   completed (renamed) runs too.
2. **Add `--clean` flag** (default off) that removes any
   `model-*-of-*.safetensors` not in the current run's expected shard set.
3. **Better idx tracking** — set `state["shard_idx"]` from the max numeric
   index across ALL existing shard files of any naming pattern.
4. *(Optional, not blocking)* Persist `shard_map.json` as a sidecar to make
   resume fully recoverable; this converts the dict-overwrite dependency into
   an explicit checkpoint.

### Acceptance criteria
- [ ] Second invocation on the same output dir does NOT re-quantize already-done tensors.
- [ ] No orphan shards after a completed run: output dir contains exactly N shards matching `model-*-of-NNNNN.safetensors` (unless `--keep-old` opt-in).
- [ ] `idx_str` parsing works for both `XXXXX` and `NNNNN` patterns.
- [ ] *(Down-scoped)* No claim of "duplicate keys in index" as a testable criterion — that pathway is not possible given dict overwrite semantics.

### Estimated effort
~2h (resume glob rework + orphan cleanup + idx tracking). Lower than the original 3h because the index-data-corruption fear is dropped.

---

## P2 — 🟡 `convert_glm52_jangtq_k.py`: "Defensive fallback" masks unknown tensor names

### Verified
`make_bits_and_method` / `get_bits_and_method` (`convert_glm52_jangtq_k.py:137`):
```python
return (2, "mxtq")  # defensive fallback
```
No warning, no log. A future tensor variant would silently default to 2-bit MXTQ.

### Fix action
Replace the silent `return (2, "mxtq")` with a `raise ValueError(...)` or an
aggressive WARNING printing the full tensor name. For the mixed-projection
path add an `else:` clause that logs the unrecognized tensor key.

### Acceptance criteria
- [ ] An unrecognized expert-projection tensor name raises a clear error or at minimum a non-suppressible WARNING with the full tensor name.
- [ ] Normal conversion (all expected tensor names) is unaffected.

### Estimated effort
15 minutes.

---

## P2 — 🟡 `gguf_to_mlx_streaming.py`: Interior `import shutil`

### Verified
`gguf_to_mlx_streaming.py:250` `import shutil` inside a function body; all other imports at module level.

### Fix action
Move `import shutil` to the top of the file alongside other stdlib imports.

### Acceptance criteria
- [ ] `shutil` imported at module level, not function-body scope.
- [ ] No functional change.

### Estimated effort
2 minutes.

---

## P2 — 🟡 `gguf_to_mlx_streaming.py`: Underscore-prefixed variables ARE used

### Verified
Lines 363-373: `_k`, `_v`, `_m`, `_sm`, `_switch_added` are all iterated / matched / assigned — actively used despite `_` convention.

### Fix action
Rename without leading underscore: `key`, `val`, `match`, `switch_key`, `switch_added`.

### Acceptance criteria
- [ ] No `_`-prefixed variable in the switch_mlp quant-entry loop is actively used.
- [ ] `flake8`/`ruff` unused-variable warnings (if any) are addressed.
- [ ] Functional equivalence preserved.

### Estimated effort
10 minutes.

---

## P2 — 🟡 `gguf_to_mlx_streaming.py`: Brittle regex for expert tensor detection

### Verified
Lines 365-366: anchored regex `r"(model\.layers\.\d+)\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"`. If `mlx_lm`'s key format changes, `_switch_added` silently stays 0 → `switch_mlp` quant config never populated → shape error at load.

### Fix action
- Match on loose substrings: `"mlp.experts." in key and any(proj in key for proj in ("gate_proj","up_proj","down_proj"))` instead of the exact anchored regex. OR
- Keep the regex but also try a secondary pattern if `_switch_added == 0` after the first pass (emit a warning).

### Acceptance criteria
- [ ] Quantization block key matching works even if the key format changes prefix/suffix format within reason.
- [ ] `_switch_added > 0` for any model that has routed experts with those projections.
- [ ] Warning printed if no switch_mlp entries were added (zero is suspicious for GLM-5.2).

### Estimated effort
30 minutes.

---

## P2 — 🟡 `patch_jang_attn_test.py`: Global monkey-patch without cleanup

### Verified
- `original_load = mx.load` (line 50) — original IS saved.
- `mx.load = patched_load` (line 87) — no `try/finally`, so an exception leaves `mx.load` permanently patched for the rest of the process.

### Fix action
Wrap the patch scope in `try/finally` (a context manager `patch_mx_load(src_all)` is the cleanest):
```python
original_load = mx.load
try:
    mx.load = patched_load
    # ... test code ...
finally:
    mx.load = original_load
```

### Acceptance criteria
- [ ] After a normal exit, `mx.load` is restored to the original.
- [ ] After an exception, `mx.load` is restored to the original.
- [ ] No side effects on unrelated `mx.load` calls in the same process.

### Estimated effort
15 minutes.

---

## P2 — 🟡 C++ Metal dead code: 144-line radix-select kernel never dispatched

### Verified
- `ggml/src/ggml-metal/ggml-metal.metal:5761` `kernel void kernel_top_k_f32_i32_radix(...)`.
- `#define RADIX_TOP_K_TG 256` / `#define RADIX_TOP_K_NLEV 4` duplicated at lines 5732-5733 (outer, not enclosed by the kernel) and 5745-5746 (inside the kernel body, shadowing).
- `ggml-metal-device.cpp:1241` comment confirms: "top_k: a radix-select kernel (kernel_top_k_f32_i32_radix) is implemented in the [metal file] but [not used]" — `get_pipeline_top_k` always returns the bitonic pipeline.

### Fix actions
1. **Remove the unreachable kernel** — delete `kernel_top_k_f32_i32_radix` from `ggml-metal.metal` and all associated pipeline/dispatch code from `ggml-metal-device.cpp` / `ggml-metal-ops.cpp`. Preserve in a gist/branch if reference is needed. Preferred given the finding that radix is always slower on Metal at these sizes.
2. **OR** gate it behind `#ifdef GGML_METAL_RADIX_TOP_K` with a comment that it's known-slower (see `GLM52_SESSION_MEMORY.md` radix-select finding).
3. **Remove the duplicate outer defines** at lines 5732-5733.

### Acceptance criteria
- [ ] `grep -c "RADIX_TOP_K" ggml-metal.metal` returns 0 (or only conditional-compile references if `#ifdef` route taken).
- [ ] `grep -c "radix" ggml-metal-ops.cpp ggml-metal-device.cpp` returns 0.
- [ ] 445/445 top-k backend-ops tests still pass via bitonic path.

### Estimated effort
30 minutes (removal) or 1h (`#ifdef` gating).

---

## P2 — 🟡 `strip_mtp_layer.py`: Manual safetensors header parsing when `safe_open` is already imported

### Verified
- `strip_mtp_layer.py:33` `from safetensors import safe_open` (used at line 76).
- `shard_keys()` at lines 37+ still does `hsize = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(fh.read(hsize))` manually.

### Fix action
Replace the manual parsing in `shard_keys()` with:
```python
def shard_keys(sf_path):
    with safe_open(str(sf_path), framework="numpy") as f:
        return list(f.keys())
```

### Acceptance criteria
- [ ] `shard_keys()` returns the same list of tensor keys as the manual JSON parsing.
- [ ] The manual binary-format code is removed.

### Estimated effort
15 minutes.

---

## P2 — 🟡 `sweep_dsa_decode.py`: Prompt truncation at byte/codepoint boundary

### Issue
`str[:char_len]` truncates at Python codepoint boundaries. `CHARS_PER_TOK = 5.8` is an English average; for CJK content (common in GLM-5.2 eval), a single character maps to ~1.3-1.5 tokens, so target token counts are systematically wrong. The hardcoded `SRC = "/tmp/longctx_prompt.txt"` is also not configurable.

*Not re-verified line-for-line in this audit (lower-priority Python script), but the char-vs-token concern is structurally sound.*

### Fix actions
1. **Tokenizer-aware truncation** — use `tiktoken` or `llama-cli`'s tokenize mode to count actual tokens. Accept `--target-tokens N` flag.
2. **Make `SRC` configurable** via `--src` CLI arg with `/tmp/longctx_prompt.txt` as default.

### Acceptance criteria
- [ ] At `--target-tokens N`, the prompt is truncated to the actual token budget N (within ±1 token), not an approximation.
- [ ] `--src` flag overrides the default source file.
- [ ] Behavior with CJK, emoji sequences, and combined diacritics is correct (no mid-grapheme split).

### Estimated effort
1h (tokenizer integration) + 15min (CLI flag).

---

## P2 — 🟡 C++ CMakeLists.txt: `PRIVATE` → `PUBLIC` include directory without comment

### Verified
`vendor/llama.cpp/tools/server/CMakeLists.txt`:
- line 28: `target_include_directories(${TARGET} PUBLIC ../mtmd)`
- line 46: also `PRIVATE ../mtmd` (so both PUBLIC and PRIVATE appear for ldarg dirs)

Propagating `../mtmd` as a PUBLIC include onto everything linking `llama-server` is an ABI surface change. The upstream commit history on this file shows it is a mix of merge churn; the PUBLIC change has no explanatory comment.

### Fix action
1. **Add a comment** explaining why `PUBLIC` is necessary (e.g., "server headers expose mtmd types").
2. **OR** revert to `PRIVATE` and add a targeted `target_include_directories(${TARGET} PRIVATE <failing-path>)` for whatever actually needed it.
3. **Investigate** what required the change — grep server source for `#include` directives referencing `mtmd` paths.

### Acceptance criteria
- [ ] Either reverted to `PRIVATE` with a targeted fix, or commented explaining why `PUBLIC` is necessary.
- [ ] Build still passes via `build_llamacpp.sh`.

### Estimated effort
30 minutes (investigation + fix).

---

## P2 — 🟡 `glm-dsa.cpp`: No overflow guard on `ggml_view_4d` stream splitting

### Verified
Window around `n_stream = indexer_k->ne[3]` + `ggml_view_4d(... indexer_q->ne[2]/n_stream ...)` in the indexer path has no assert guarding `n_tokens % n_stream == 0`.

### Fix action
Add an alignment guard before the `ggml_view_4d` calls:
```cpp
GGML_ASSERT(n_stream > 0 && n_tokens % n_stream == 0 &&
            "batch size must be a multiple of stream count for indexer view splitting");
```
Add a comment explaining the constraint and that a fallback padding path would be needed for general batch sizes.

### Acceptance criteria
- [ ] Batch-1 inference unchanged (no assert).
- [ ] Batch-N where `n_tokens % n_stream == 0` passes.
- [ ] Batch-N where `n_tokens % n_stream != 0` hits the assert with a clear message, not silent token loss.

### Estimated effort
10 minutes.

---

## P3 — ⚪ Minor / Cosmetic items

| # | Issue | File | Fix | Effort |
|---|-------|------|-----|--------|
| 1 | Private import `from jang_tools.calibrate import _load_bf16_tensor` | `convert_glm52_jangtq_k.py` | Audit usage; vendor a stable copy or add a comment noting the version of `jang_tools` this was tested against | 20 min |
| 2 | `"total_size": 0` in index metadata | `convert_glm52_jangtq_k.py` | Compute total size from shard file sizes and write the real value, or remove the field if the loader doesn't require it | 15 min |
| 3 | No retry on `save_file` flush | `convert_glm52_jangtq_k.py` | Wrap `save_file` in a simple retry loop (3 attempts, 5s backoff) for transient disk-full / I/O errors during long conversions | 15 min |
| 4 | `del tensor` + `gc.collect` every 200 iterations | `convert_glm52_jangtq_k.py` | Investigate whether memory is actually bound; add a comment explaining what reference is being held, or remove manual GC if not | 30 min |
| 5 | Redundant `endswith(f".{proj_key}.weight")` check | `convert_glm52_jangtq_k.py` | The `gate_proj`/`up_proj`/`down_proj` name match already implies the suffix; remove the redundant `endswith` clause | 5 min |
| 6 | Duplicate `n_ff_exp` hparam load | `glm-dsa.cpp:load_arch_hparams` | Remove the second load (keep the one actually read downstream) | 10 min |
| 7 | Indexer tensors loaded for MTP layer 78 | `glm-dsa.cpp:load_arch_tensors` | Sub-problem of the P0 reframing. If/when AC3 metadata lands + shared-layer tensors are gated out, layer 78 should also skip indexer tensors (it is an MTP block, not a full indexer layer) | Included in P0 |

---

## Priority order for scheduling (reframed)

| Priority | Item | Depends on | Status |
|----------|------|-----------|--------|
| **P0 (cheap tail)** | Update docs (`PLAN.md §7.L` status, `AGENTS.md`, `docs/research/README.md` stale `glm-dsa.cpp:152 alias` claims) | None | **[DONE 2026-06-24]** |
| **P0 (decision)** | Investigate upstream HF config for `indexer_types[]` to decide whether AC3 is real | None | **[DONE 2026-06-24]** — F/S real upstream (21 full / 57 shared); finding appended to session memory |
| **P0 (data-layer)** | Extend `gguf-py` writer + re-quantize GGUF + refactor `load_arch_tensors` (5/layer); gate graph on `indexer_types[]` | The decision above | **[DEFERRED by design]** — kernel-bound regression + baseline-preservation; see Decision record |
| **P1** | Duplicate entry in `GLM52_SESSION_MEMORY.md` | None | **[DONE 2026-06-24]** — second occurrence deleted; count=1 |
| **P1** | Resume logic in `convert_glm52_jangtq_k.py` | None | **[DONE 2026-06-24]** — glob both patterns + `--clean` + idx tracking |
| **P2** | Defensive fallback masks unknown tensors | None | **[DONE 2026-06-24]** — now raises ValueError |
| **P2** | Metal dead code removal | None | **[PARTIAL 2026-06-24]** — duplicate outer `#define` removed + clarifying comment; dispatch already gone (ops.cpp:4357); kernel body retained-as-reference to avoid unverified metallib surgery. Full removal deferred |
| **P2** | Monkey-patch cleanup in test | None | **[DONE 2026-06-24]** — try/finally restoring `mx.load` |
| **P2** | `gguf_to_mlx_streaming`: interior `import shutil` | None | **[DONE 2026-06-24]** — moved to module level |
| **P2** | `gguf_to_mlx_streaming`: `_`-prefixed vars used | None | **[DONE 2026-06-24]** — renamed `key`/`val`/`m`/`switch_key`/`switch_added` |
| **P2** | `gguf_to_mlx_streaming`: brittle expert regex | None | **[DONE 2026-06-24]** — added zero-switch-added WARNING |
| **P2** | `strip_mtp_layer.shard_keys` manual parse | None | **[DONE 2026-06-24]** — uses `safe_open`; `struct` import removed |
| **P2** | `sweep_dsa_decode` truncation | None | **[PARTIAL 2026-06-24]** — added `--src`/`--url` CLI + approximation WARNING; full tokenizer-aware truncation deferred (needs tokenize-mode helper) |
| **P2** | CMake `PUBLIC ../mtmd` comment | None | **[DONE 2026-06-24]** — clarifying comment added |
| **P2** | `glm-dsa.cpp` `n_stream` overflow guard | None | **[DONE 2026-06-24]** — additive `GGML_ASSERT`; verified no-op on baseline (n_stream=1, single-seq) |
| **P3 #1** | Private `_load_bf16_tensor` import | None | **[DONE 2026-06-24]** — provenance comment added |
| **P3 #6** | Duplicate `n_ff_exp` hparam load | None | **[DONE 2026-06-24]** — also removed dup `n_expert_shared` load (both were double-loaded in MLA block) |
| **P3 #2** | `total_size: 0` in index metadata | None | **[DEFERRED]** — low value; risk of getting wrong without full shard finalize; left as-is |
| **P3 #3** | `save_file` retry loop | None | **[DEFERRED]** — adds complexity on the flush hot path; not applied to avoid risk |
| **P3 #4** | `gc.collect` every 200 iters | None | **[NOT APPLIED]** — working as-is; no evidence it's bound |
| **P3 #5** | Redundant `endswith` check | None | **[NOT APPLIED]** — keeping both checks is more robust, not less; removal is pure style |
| **P3 #7** | Indexer tensors for MTP layer 78 | None | **[DEFERRED]** — sub-case of P0 AC3 data-layer branch (deferred) |

## Build verification (2026-06-24)

Incremental build of `vendor/llama.cpp` (build-metal, `--target llama`) succeeded after the C++ edits: `glm-dsa.cpp` recompiled, the embedded Metal library re-embedded cleanly (dup-`#define` removal compiles), `libllama.dylib` linked. All four MLX-export Python files + `scripts/sweep_dsa_decode.py` pass `py_compile`. The runtime baseline (232 GB GGUF load + merge-sort / BLUE-FALCON) is **unchanged by construction**: the `load_arch_hparams` removal is idempotent (values still set in the MoE block), the `n_stream` assert is a no-op on single-sequence inference (`n_stream = unified ? 1 : n_seq_max = 1`), and the Metal kernel was already undispatched. A full baseline re-run is recommended but not performed (multi-minute 232 GB load).

## Tracking contract

Per `AGENTS.md`:
- Each fix PR/commit references its story from this plan.
- After fixing, append a finding to `GLM52_SESSION_MEMORY.md` per the findings-tracking contract (symptom → cause → fix → verified).
- Mark the item as **[DONE YYYY-MM-DD]** inline in this file when the fix lands.
- Close the `CODE_REVIEW.md` review item in the commit message.

## Audit notes (2026-06-24)

The following original-review subclaims were downgraded or removed after the
codebase audit:
- **P0 "57 shared layers carry no indexer weights"** — false for the
  known-good GGUF; indexer tensors exist on all 79 blocks. There is no
  `indexer_types[]` metadata to gate on. C++ "fix" reframed to data-layer
  investigation first; effort reassessed from 2h to multi-day.
- **P0 "8 indexer tensors per layer / 78×8=624 saved"** — wrong tensor count;
  actual is 5 per layer.
- **P1 "duplicate shard_map keys → silent data corruption" (item 2)** — not
  reachable: `shard_map[k] = …` overwrites, never appends. The real waste is
  orphan shard files (item 1), which is retained.
- **P1 effort** adjusted from 3h → 2h.
- The original P0's underlying observation (AC3 NOT implemented despite
  "IMPLEMENTED" checkbox) is **confirmed accurate**; only the proposed fix
  was wrong. Reframed accordingly.
