# Code Review — Roast of the Kitchen Colleague

Review based on `git diff HEAD` (uncommitted changes) + `git diff HEAD~1..HEAD` (last commit), covering 9 files across Python, C++, Metal, and documentation.

---

## 🔴 False Acceptance Criteria (AC3 — PLAN.md §7.L)

**File:** `PLAN.md`
**Status:** `IMPLEMENTED 2026-06-23 (AC1-AC6 done)`
**Reality:** NOT implemented.

PLAN.md §7.L AC3 says:

> **AC3** — F/S IndexShare: the GLM-DSA graph runs the lightning indexer + top-k ONLY on "full" layers (where `indexer_attn_q_b != nullptr`); "shared" layers reuse the most recent full layer's top-k indices. No null-tensor deref on the 57 shared layers.

Now look at `vendor/llama.cpp/src/models/glm-dsa.cpp`, graph constructor:

```cpp
for (int il = 0; il < n_layer; ++il) {
    ...
    ggml_tensor * indexer_q = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_q_b, qr);
    ...
```

**Every layer runs the indexer.** There is zero layer-role gating. No `if (layer_is_full)` / `else { reuse previous top_k }`. No F/S distinction at runtime. The comment in `models.h` even concedes:

> The GLM-5.2 GGUF carries a full indexer on every block (no F/S sharing at the GGUF level), so the body mirrors `llama_model_deepseek32::graph`.

`load_arch_tensors` loads ALL 8 indexer tensors for ALL `n_layer_all` layers (including MTP layer 78). The 57 "shared" layers' indexer weights are dead weight in RAM (~200 GB) — loaded but conceptually unnecessary.

The AC says "only on full layers." The code says "on every layer." **Checkbox painted green when it's empty.**

---

## 🟠 Docs: Duplicate entry in GLM52_SESSION_MEMORY.md

**File:** `GLM52_SESSION_MEMORY.md` (uncommitted, 507 lines added)

Two entries with **identical title and date**:

```
### 2026-06-23 (C) — DSA context-length sweep: crossover NOT reachable with full-sort top_k
```

Then immediately repeated:

```
### 2026-06-23 (C) — DSA context-length sweep: crossover NOT reachable with full-sort top_k
```

Same content, same date tag, same data table. One is a copy-paste ghost. The findings record — which the repo contract (AGENTS.md) says *must* be the canonical durable memory — now has a factual tachyon. Which copy is authoritative?

---

## 🟠 `convert_glm52_jangtq_k.py`: Resume logic is theater

**File:** `mlx-export/convert_glm52_jangtq_k.py`

### Problem 1: Resume glob can't find completed shards

```python
existing = sorted(OUT.glob("model-*-of-XXXXX.safetensors"))
```

After a **completed** run, the rename step turns all `XXXXX` → `NNNNN`:

```python
old = OUT / f"model-{i:05d}-of-XXXXX.safetensors"
new = OUT / f"model-{i:05d}-of-{n_shards:05d}.safetensors"
if old.exists():
    old.rename(new)
```

Next invocation: `done_keys` is empty → re-does **every tensor from scratch**, writes into new shards with new indices. Old `model-*-of-NNNNN.safetensors` files are **never cleaned up** — orphan garbage accumulating on disk. Three runs = 3× the disk usage, all but one set unreferenced.

### Problem 2: Stale index entries across resumes

If a partial run writes shards with `XXXXX`, then crashes, the shard_map records tensors → shard filename. On resume, existing `XXXXX` shards' tensors are added to `shard_map`. If the converter re-quantizes a tensor (because its `tq_packed`/`tq_norms`/`tq_bits` triad was incomplete), **both the old (orphan) and new entries** land in the final index pointing at DIFFERENT shard files for the SAME tensor key. The loader reads the first via the index — silent data corruption vector.

### Problem 3: idx_str parsing

```python
idx_str = sf.name.split("-")[1]
state["shard_idx"] = max(state["shard_idx"], int(idx_str))
```

`state["shard_idx"]` starts at 0. If existing shards max out at index 3, the next flush writes shard 4. Correct. But after the rename, next run has no `XXXXX` shards → state stays 0 → fresh write + gap in shard numbering. Functional but messy.

---

## 🟡 `convert_glm52_jangtq_k.py`: "Defensive fallback" hides bits assignment bugs

```python
def get_bits_and_method(tensor_name):
    ...
    if "experts" in name and ("gate_proj" in name or "up_proj" in name or "down_proj" in name):
        if mixed_proj_bits is not None:
            for proj_key, proj_bits in mixed_proj_bits.items():
                if f".{proj_key}." in name or name.endswith(f".{proj_key}.weight"):
                    return (proj_bits, "mxtq")
            return (2, "mxtq")  # defensive fallback ← HERE
```

If an expert projection doesn't match `gate_proj`, `up_proj`, or `down_proj` (typo, future variant, or a renamed tensor), the code **silently** assigns 2-bit MXTQ with zero log/warning/error. A fallback should **explode** on unexpected tensor names, not silently corrupt a weight you didn't know existed.

---

## 🟡 `gguf_to_mlx_streaming.py`: `import shutil` inside function body

```python
if reference_tokenizer:
    import shutil
```

`shutil` is in stdlib. Import cost is negligible. But every other import in the file is at module level. This import is inline because the feature was bolted on after the function was written and nobody moved it up. Code smell.

---

## 🟡 `gguf_to_mlx_streaming.py`: Underscore-prefixed variables that ARE used

```python
for _k, _v in list(quantization_block.items()):
    _m = re.match(
        r"(model\.layers\.\d+)\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",
        _k,
    )
    if _m and isinstance(_v, dict):
        _sm = f"{_m.group(1)}.mlp.switch_mlp.{_m.group(2)}"
        if _sm not in quantization_block:
            quantization_block[_sm] = _v
```

Python convention: `_` prefix = "unused." These ARE used — iterated, matched, grouped, assigned. The underscore tells the reader "I don't care about this value," but the code clearly does. Rename them without `_` or restructure.

---

## 🟡 `gguf_to_mlx_streaming.py`: Brittle regex for detecting expert tensors

```python
r"(model\.layers\.\d+)\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
```

Assumes the quantization block keys follow exactly this pattern. If `mlx_lm`'s internal key format changes (e.g. adds a prefix, flattens the structure, changes `experts` pluralization), this silently matches nothing, `_switch_added` stays 0, and the switch_mlp quant config is never populated → shape error at load time. A structural mapping from per-expert → stacked would be more resilient.

---

## 🟡 `patch_jang_attn_test.py`: Global monkey-patch with no cleanup

```python
original_load = mx.load
...
mx.load = patched_load   # installs global hook
```

The patched `mx.load` closes over `src_all` — a dict populated from a completely different file scan. If `load_model()` or any downstream code calls `mx.load` for any unrelated purpose, it routes through the patcher referencing a stale dict. No `try/finally: mx.load = original_load` — any exception leaves `mx.load` permanently broken for the entire Python process.

---

## 🟡 C++ Metal dead code: 144-line radix-select kernel, never dispatched

**Files:** `ggml/src/ggml-metal/ggml-metal.metal`, `ggml/src/ggml-metal/ggml-metal-ops.cpp`, `ggml/src/ggml-metal/ggml-metal-device.cpp`

```cpp
// top_k: a radix-select kernel (kernel_top_k_f32_i32_radix) is implemented in the
// .metal file but is NOT selected here
```

The Metal kernel is 144 lines of code. The C++ dispatch site:

```cpp
auto pipeline = ggml_metal_library_get_pipeline_top_k(lib, op);
// ---- bitonic argsort + merge path (default) ----
```

`get_pipeline_top_k` returns the bitonic pipeline. The radix kernel is **never compiled into the function pipeline table**. No `#ifdef` to toggle it. No compile flag. Just 144 lines of unreachable code permanently occupying `ggml-metal.metal`. If you want it for reference, keep it in a branch or an external gist. Don't commit dead code to the working tree where the next developer spends 15 minutes tracing the call graph before discovering it goes nowhere.

**Also:** the radix kernel has `#define RADIX_TOP_K_TG 256` and `#define RADIX_TOP_K_NLEV 4` — **lines 5728 and 5732 are identical defines.** The first one (line 5728) is outside the function body with a typedef — the second (line 5732) is inside the function body shadowing it. Unused after that, but it looks like a mangled refactor.

---

## 🟡 `strip_mtp_layer.py`: Manual safetensors header parsing when `safe_open` is already imported

```python
def shard_keys(sf_path):
    with open(sf_path, "rb") as f:
        hsize = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hsize))
    return [k for k in hdr if k != "__metadata__"]
```

`safe_open(str(sf_path), framework="numpy").keys()` is two lines and doesn't duplicate the safetensors binary format spec. You already `from safetensors import safe_open` at the top of the file. Why hand-roll a format parser that's already in your import table?

---

## 🟡 `sweep_dsa_decode.py`: Prompt truncation at byte boundary, no tokenizer

```python
char_len = int(tgt * CHARS_PER_TOK)
prompt = text[:char_len]
```

`str[:N]` operates on code points, not tokens. `CHARS_PER_TOK = 5.8` is an English average. For CJK content (common in GLM-5.2 eval), a single character maps to ~1.3-1.5 tokens, so the target token counts are **systematically wrong**. Also, slicing at a random code-point boundary can land in the middle of a grapheme cluster (emoji sequences, combined diacritics). A `tiktoken` call costs one import. The hardcoded `SRC = "/tmp/longctx_prompt.txt"` is also not configurable.

---

## 🟡 C++ CMakeLists.txt: `PRIVATE` → `PUBLIC` include directory with no explanation

```cmake
-target_include_directories(${TARGET} PRIVATE ../mtmd)
+target_include_directories(${TARGET} PUBLIC ../mtmd)
```

`PUBLIC` propagates the `../mtmd` include path to every library that links `llama-server`. This is namespace pollution — anything downstream of `llama-server` now inherits mtmd headers. If the linker ever resolves the wrong `llama.h` from mtmd instead of the canonical one, the resulting segfault trace will take a day to debug. What build error was this fixing? No comment.

---

## 🟡 C++ `glm-dsa.cpp`: No overflow guard on `ggml_view_4d` stream splitting

```cpp
const auto n_stream = indexer_k->ne[3];
indexer_q = ggml_view_4d(ctx0, indexer_q,
    indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream,
    ...);
```

For batch-1 inference, `n_stream = 1`, division is safe. For batch-N where `n_tokens % n_stream != 0`, integer division **truncates** the last partial batch's tokens. No assert, no padding, no alignment guard. Silent token loss in multi-batch decode.

---

## ⚪ Minor / Cosmetic

| Issue | File | Detail |
|-------|------|--------|
| Private import | `convert_glm52_jangtq_k.py` | `from jang_tools.calibrate import _load_bf16_tensor` — fragile against library refactors |
| `total_size: 0` | `convert_glm52_jangtq_k.py` | Index metadata `"total_size": 0` — placeholder value never filled |
| No retry on flush | `convert_glm52_jangtq_k.py` | `save_file` on a full shard has no error handling for disk-full / I/O errors |
| `del tensor` + `gc.collect` every 200 iterations | `convert_glm52_jangtq_k.py` | Manual GC is a smell — either the reference is held somewhere unexpected, or memory isn't the problem |
| `@` prefix on query in sweep | `sweep_dsa_decode.py` | `endswith(f".{proj_key}.weight")` check — redundant if the name already matched `gate_proj`/`up_proj`/`down_proj` earlier |
| Duplicate `n_ff_exp` hparam load | `glm-dsa.cpp:load_arch_hparams` | Loaded twice — once at line 13, once at line 33 |
| Indexer tensors loaded for MTP layer 78 | `glm-dsa.cpp:load_arch_tensors` | All 78 layers' indexers are loaded, but the graph only uses 0–77. MTP layer 78's indexer tensors are dead weight in RAM (~200 GB). |

---

## Verdict

The real engineering wins here — DSA sparse-attention graph patched to actually RUN on GLM-5.2, the JANGTQ_K converter producing a working coherent 279 GB MLX bundle, the 100K-context needle retrieval passing — are substantial. But they're undermined by:

- **Acceptance criteria marked done when the code doesn't match.** AC3 (F/S layer gating) is not just incomplete — it's not even started. PLAN.md's "IMPLEMENTED" claim is misleading.
- **Critical utility functions (resume) broken by design.** The conversion tool's resume feature globs for `XXXXX` shards that don't exist after a completed run. It silently redoes everything and leaves orphan garbage.
- **Dead code committed to Metal source.** 144 lines of an unreachable radix-select kernel with no compile-time gate to activate it.
- **Global monkey-patching without cleanup or guards.**
- **Duplicate documentation** that fractures the canonical findings record.
