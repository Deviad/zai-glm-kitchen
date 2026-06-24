# GLM-5.2 Kitchen — Test Results

Hardware: Apple M3 Ultra, 512 GB unified memory. All tests 2026-06-23.
Chat mode, OpenAI-compatible `/v1/chat/completions`.

GLM-5.2 emits chain-of-thought in the `reasoning_content` field and the final
answer in `content`. With thinking ON, short prompts can exhaust `max_tokens`
inside reasoning before `content` is written — pass
`chat_template_kwargs={"enable_thinking":false}` for direct answers.

---

## Models under test

| Tag | Format | Serve | Bundle | RSS | Notes |
|---|---|---|---|---|---|
| **GLM-5.2-JANGTQ_K** | MLX (TurboQuant, mixed bits) | `vmlx-engine` :8080 | 279 GB | ~260 GB | experts gate/up 2-bit + down 4-bit MXTQ; attn/shared/embed/head fp16; full 78 layers |
| **GLM-5.2-shortgpt-pruned-IQ2S-IQ4NL** | GGUF (IQ2_S experts / IQ4_NL rest) | `llama-server` :8081 | 189 GB | 189 GB | shortgpt layer-pruned; patched llama.cpp Metal build |

JANGTQ_K bundle: `/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-JANGTQ_K/` (277 shards,
~3.51 bpw, tq_bits histogram `{2: 38912, 4: 19456}`, zero 1-bit tensors).
GGUF: `/Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-shortgpt-pruned-IQ2S-experts-IQ4NL-rest/`
(9 shards, 191 GB on disk).

---

## 1. Short-prompt throughput shootout (≤8K context)

Same prompt battery, temperature 0.1–0.3.

| Test | JANGTQ_K (MLX) | shortgpt GGUF | Both correct? |
|---|---|---|---|
| 17 × 23 | ~9 tok/s → **391** | 24.7 tok/s → **391** | ✓ |
| `is_prime(n)` | ~9.5 tok/s, 6k±1 wheel | 24.6 tok/s, `math.isqrt` | ✓ |
| 3 largest planets (think off) | ~8.4 tok/s → Jupiter/Saturn/Uranus | 24.4 tok/s → Jupiter/Saturn/Uranus | ✓ |
| Capital of Japan | ~9.5 tok/s → "Tokyo." | (n/a) | ✓ |
| Multi-turn "42" recall | 42×2=84 even | 42×2=84 even | ✓ |
| Photosynthesis 150w (think off) | ~9.8 tok/s, accurate | 22.7 tok/s, accurate | ✓ |

**Server-measured generation throughput:**

| Model | Gen tok/s | Prompt tok/s |
|---|---|---|
| GLM-5.2-JANGTQ_K (MLX) | ~8.4–10.0 | — |
| GLM-5.2-shortgpt GGUF | **24.48** | 43.9 |

**Winner (throughput): GGUF shortgpt-pruned IQ2_S — ~2.4× faster.**
Both fully coherent, zero gibberish (with thinking off).

Why GGUF wins: (a) shortgpt layer pruning removes whole transformer layers →
fewer matmuls/token; (b) llama.cpp IQ2_S/IQ4_NL Metal kernels are highly
optimized; (c) smaller bundle (191 vs 279 GB) → less memory bandwidth/token.
The MLX JANGTQ_K keeps attention/shared/embed at fp16 and runs all 78 layers.

---

## 2. Long-context realistic scenario — 100K window, ~50K-token prompt

**Model:** GLM-5.2-shortgpt-pruned-IQ2S-IQ4NL (GGUF), `llama-server`.
**Server config:** `-c 102400` (100K context), `-np 1` (single slot),
`-fa on` (flash attention), `-ngl 99`.

**Workload:** needle-in-a-haystack. A 700-section synthetic company knowledge
base with one unique internal record buried at section 350 (≈50% depth). Prompt
tokenized to **53,633 tokens**. Question asks to retrieve the passphrase, the
responsible team, and the rotation interval.

**Retrieval result — PERFECT (3/3 facts):**

> *"The passphrase is BLUE-FALCON-48217, which the platform security team must
> rotate every 90 days."*

| Fact | Expected | Retrieved |
|---|---|---|
| Passphrase | BLUE-FALCON-48217 | ✓ |
| Team | platform security team | ✓ |
| Interval | every 90 days | ✓ |

`finish_reason=stop`, 25 completion tokens, no hallucination.

**Timing (53.6K-token prompt):**

| Phase | Rate | Notes |
|---|---|---|
| Cold prefill (uncached) | 233 tok/s → ~29 tok/s | O(n²) attention decay; starts fast, slows as context fills. ~1085 s wall for full 53.6K. |
| Cached prefill (re-run) | instant (prompt-cache hit) | 0.2 s |
| Generation | ~7.4–7.5 tok/s | decode slower than short-ctx 24 tok/s because each token attends over 53.6K KV |

Cold-prefill decay samples (uncached):

| Context filled | Prefill tok/s |
|---|---|
| 2,050 | 233.1 |
| 10,242 | 129.9 |
| 20,482 | 92.1 |
| 30,722 | 71.7 |
| 36,866 | 63.4 |
| ~53,600 (tail) | ~29 |

**Decode vs context (same model, same hardware):**

| Context | Decode tok/s | Note |
|---|---|---|
| ≤8K | 24.5–25.4 | short-prompt baseline |
| ~54K (f16 KV) | **7.2** | sustained over 160 tok |
| ~54K (q8_0 KV) | 2.2 | KV quant HURTS on Metal — dequant overhead per step > bandwidth saved; reverted |

The decode slowdown from 24.5 → 7.2 tok/s is inherent: every generated token
attends over the full ~54K KV cache. This is expected MLA/transformer behavior,
not a quantization defect. Any tok/s claim must state its context length —
comparing a short-context number on model A against a 54K-context number on
model B is not apples-to-apples.

**Operational notes:**
- Client timeouts must be generous (>20 min) for cold 50K prefill, OR rely on
  llama.cpp's prompt cache: a cancelled request still caches its partial
  prefill, so a resend resumes from the cached prefix (observed: run 2 started
  at 76% progress after run 1 was cancelled at 69%).
- Flash attention (`-fa on`) and single-slot (`-np 1`) give the full 100K window
  to one request and reduce KV memory pressure.
- RSS stayed ~198 GB at 53.6K context (model 189 GB + KV/compute) — well within
  512 GB. MLA (kv_lora_rank>0) keeps the KV cache compact.

---

## Recommendations

- **Throughput / interactive / long-context serving → GGUF shortgpt-pruned IQ2_S.**
  ~2.4× faster decode than MLX at short context, smaller footprint, correct
  50K-token retrieval, runs on the patched llama.cpp Metal build.
- **Keep KV cache at f16** — q8_0 KV is 3.3× slower at long context on Metal.
- **Always cite context length with any tok/s number** — this model is 24.5
  tok/s at ≤8K and 7.2 tok/s at 54K; both are the same model.
- **Maximum fidelity headroom → MLX JANGTQ_K.** Full 78 layers, fp16 attention/
  shared/embed/head, no aggressive 2-bit layer pruning — trades speed for
  quality margin. Served via the vmlx-engine JANGTQ native path.

---

## Reproduction

**vmlx-engine (MLX JANGTQ_K):**
```
/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/vmlx-engine serve \
  "/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-JANGTQ_K" --host 127.0.0.1 --port 8080
```

**llama-server (GGUF), 100K context:**
```
vendor/llama.cpp/build-metal/bin/llama-server \
  -m "/Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-shortgpt-pruned-IQ2S-experts-IQ4NL-rest/GLM-5.2-shortgpt-pruned-IQ2_S-IQ4_NL-00001-of-00009.gguf" \
  --host 127.0.0.1 --port 8081 --jinja -ngl 99 -c 102400 -np 1 -fa on --metrics
```

Logs:
- `logs/vmlx_serve_jangtq_k_20260623_193615.log`
- `logs/llama_server_shortgpt_20260623_194503.log` (8K ctx)
- `logs/llama_server_shortgpt_100k_20260623_195309.log` (100K ctx, 50K test)

---

## 3. Root cause of long-context decode slowdown (llama.cpp)

GLM-5.2 is a **DSA (DeepSeek Sparse Attention)** model (`index_topk: 2048`).
With DSA, decode tok/s should stay near-flat as context grows (each token
attends to ~2048 selected KV entries, not all of them). **llama.cpp does not
implement GLM-DSA sparse attention** — confirmed in the patched source:

1. `models.h:1115` — `llama_model_glm_dsa` aliases its graph to the **dense**
   `llama_model_deepseek2::graph` (no top-k, no sparsity).
2. `glm-dsa.cpp` **loads** 8 indexer tensors; `deepseek2.cpp` (the running
   graph) references the indexer **0 times** → dead weight.
3. `llama-model.cpp:~2026` — the sparse `llama_kv_cache_dsa` is created **only**
   for `LLM_ARCH_DEEPSEEK32`. **No `case LLM_ARCH_GLM_DSA`** → it falls through
   to a standard **dense** KV cache.

Therefore the 24.5 → 7.2 tok/s drop (≤8K → 54K) is **dense O(n) attention**, a
missing-implementation issue in llama.cpp — not a quantization defect, and not
fixable with KV-cache flags (q8_0 made it 3.3× worse). This is the gap
Salvatore Sanfilippo cited as the reason for Dwarf Star 4.

---

## 4. Patching llama.cpp to RUN GLM-DSA sparse attention — result

Implemented the missing GLM-DSA sparse forward path in the patched llama.cpp
(4 edits: own graph in models.h, DSA graph body in glm-dsa.cpp, GLM_DSA added to
the sparse KV-cache switch in llama-model.cpp, and the critical Hadamard
`attn_rot_k` arch-gate fix in llama-kv-cache.cpp — the indexer rotation matrix
was hardcoded to DEEPSEEK32-only, so GLM crashed with a null `self_k_rot_lid`).

**Correctness: SUCCESS.** No segfault; `creating indexer KV cache` +
`attn_rot_k = 1` confirmed in the load log; the lightning indexer + `ggml_top_k`
+ sparse `build_attn` now execute (18 DSA ops in the graph, previously 0).
`What is 2+2? Answer: 4` — coherent. The indexer tensors are no longer dead
weight.

**Performance: the honest surprise — DSA is SLOWER here, not faster.**

| Decode @ 53.6K context | tok/s |
|---|---|
| Dense (old, deepseek2 alias) | 7.2 |
| **DSA sparse (this patch)** | **4.22** |

Why: with `index_topk=2048` at 54K, each generated token must (a) score ALL 54K
cached indexer keys (an O(n) matmul per layer), (b) `ggml_top_k(54K→2048)` — a
sort-based op — then (c) attend over 2048 keys. Steps (a)+(b) cost more than the
dense attention they replace at this context length on Metal, because
llama.cpp's DSA kernels are unoptimized: no fused indexer, generic-sort top-k,
no Metal sparse-gather fast path.

**This is precisely the point behind Dwarf Star 4.** The bottleneck is the DSA
*kernel implementation* in llama.cpp, not the architecture. Making GLM-DSA run
its real sparse graph is necessary but not sufficient — the speedup only
materializes once the indexer-scoring, top-k, and sparse-gather kernels are
optimized (fused/Metal-native). The crossover would also favor sparse at much
larger contexts or smaller `index_topk`.

---

## 5. (C) DSA context-length sweep — crossover not reachable (full-sort top_k)

Patched llama.cpp, shortgpt-pruned GGUF, default `index_topk=2048`, `/completion`
n_predict=48. DSA decode tok/s vs context:

| Context | DSA decode tok/s | Dense ref |
|---|---|---|
|  ~2K | 10.08 | ~24.5 (@ <=8K) |
|  ~4K |  8.95 |       |
|  ~8K |  8.13 | 24.5   |
| ~16K |  6.96 |        |
| ~32K |  5.47 |        |
| ~54K |  4.22 |  7.2   |

Monotonic-down; DSA slower than dense at EVERY context 2K-54K; gap exists
even at 2K (10 vs ~24). `ggml_top_k` dispatches `kernel_argsort` = full bitonic
argsort over ALL n cached keys (O(n log^2 n)) regardless of k — k only shrinks
the final attention (O(k)), not the sort. So reducing `index_topk` cannot fix
this. Only a partial-select (quickselect/threshold) top-k kernel makes DSA
viable. This is the proven minimum bar before DSA beats dense.

## 6. (B) MTP self-speculative decoding — scope assessment

- MTP head EXISTS in the full mixed GGUF (blk.78, per AGENTS.md) and is LOADED
  by glm-dsa/deepseek32 (layer.nextn.{eh_proj,enorm,hnorm,shared_head_*}).
- It is NEVER EXECUTED: `deepseek32.cpp` marks it "NextN/MTP tensors
  (preserved but unused)". No forward-path code touches them for glm-dsa.
- A reference MTP forward EXISTS in `src/models/cohere2moe.cpp` (lines ~305-420):
  exactly the same tensors (eh_proj/enorm/hnorm/shared_head). Portable pattern.
- llama.cpp server speculative decode = SEPARATE DRAFT MODEL only
  (--model-draft / --spec-draft-*). NO in-model MTP / EAGLE-shared / layer-skip
  self-spec path. So in-model MTP requires either (i) refactoring the spec
  machinery to accept an in-model draft callback, or (ii) extracting the blk.78
  MTP head into a standalone draft GGUF + a minimal arch. Both are multi-day.

KEY nuance from (C): spec decode and the DSA bottleneck address DIFFERENT
regimes. Spec decode amortizes per-token weight-bandwidth (helps bandwidth-bound
decode = short-mid context, dense path). It does NOT fix the per-token
indexer+sort compute, which is the DSA long-context bottleneck (each verified
token still needs its own indexer pass). So (B) helps a different problem than
(A), which (C) proved is the necessary fix for DSA long context.

## §7 Option (A): radix-select top-k kernel — NEGATIVE RESULT (2026-06-23)

Implemented `kernel_top_k_f32_i32_radix` (Metal) replacing the blocked bitonic
argsort for `GGML_OP_TOP_K`. **Correctness: 445/445 test-backend-ops pass.**
**Performance: regressed ~20% at every context** (radix reads the row 5× from
global memory + atomic contention vs bitonic's single blocked read; for n≤54K
the bitonic is already ~O(n)).

| Context | bitonic tok/s | radix tok/s |
|---|---|---|
| 2K  | 10.08 | 7.64 |
| 4K  | 8.95  | 7.17 |
| 16K | 6.96  | 5.86 |
| 32K | 5.47  | 4.33 |

The (C) premise ("full O(n log²n) sort is the bottleneck") was imprecise —
the bitonic is already blocked (~O(n)). The real DSA long-ctx deficit is the
unavoidable `mul_mat(indexer_k, indexer_q)` over all n cached keys + sparse
gather, vs Metal's optimized dense flash-attn. No top_k kernel can fix it.
Reverted to bitonic. Radix source kept for reference.

## §8 Option (3) physical gather of top-k K/V — NEGATIVE RESULT (2026-06-24)

Implemented a gated GGML_DSA_GATHER=1 branch in `build_attn` (top_k overload)
that physically gathers the k=2048 top K/V rows before running flash-attn,
instead of running dense attention masked to -INF outside the top-k set.

**Coherence: PASS** (2+2 = 4). **Performance: 5-22% slower than bitonic
baseline at all contexts.**

| Context | Bitonic | Gather | Δ      |
|---------|---------|--------|--------|
| 2K      | 10.08   | 8.34   | -17%   |
| 4K      | 8.95    | 7.91   | -12%   |
| 8K      | 8.13    | 7.30   | -10%   |
| 16K     | 6.96    | 6.33   | -9%    |
| 32K     | 5.47    | 4.98   | -9%    |

**Why:** MLA's attention with absorption (n_head_kv=1) is already cheap on
Metal (single matmul, well-tuned flash-attn). The sparse gather saves ~32 MB
bandwidth per layer at 32K context but adds its own scatter-read+write
overhead (4 MB F32 output, then F16 cast back). Net: gather overhead
exceeds the small dense-MLA savings. Reverted; experiment code removed from
`llama-graph.cpp`.
