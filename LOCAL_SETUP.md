# Local setup

The kitchen is portable — no code paths are hardcoded. Two external
dependencies are vendored as git submodules inside `vendor/`; everything else
is owned by the kitchen itself.

## Vendored submodules

After a fresh clone, init submodules once:

```sh
git clone <kitchen-url>
cd zai-glm-kitchen
git submodule update --init --recursive      # ~5 min for the llama.cpp clone
```

| Submodule | Branch (Deviad fork) | Provides |
|---|---|---|
| `vendor/gguf2mlx` | `feature/update_for_glm5.2_cooking` (`90affca2…`) | Imported at runtime as `from gguf2mlx import convert` (declared as a uv path dep in `pyproject.toml`). End-to-end GGUF→MLX translator for the Phase 8 MLX conversion step. |
| `vendor/llama.cpp` | `feature/patch_used_to_create_mixed_quantization_of_glm5.2` (`6f67b8a…`) | Pinned source of the patched llama.cpp fork. The patched build produces `llama-cli`, `llama-quantize`, `llama-trace-moe`, `llama-tokenize`, `llama-gguf-split` from `$ROOT/vendor/llama.cpp/build-metal/bin/`. The trace-moe example + ShortGPT Block Influence computation live in `examples/trace-moe/` inside the submodule. |
| `vendor/mlx-lm` | `ml-explore/mlx-lm` main (`2c008fd…`, v0.31.3-13-g2c008fd) | Reference-only upstream `mlx-lm` (the Apple MLX LM inference library) cloned so the GLM-5.2 MLX-side code is greppable alongside the other vendored sources. The GLM-5.2 model class lives at `mlx_lm/models/glm_moe_dsa.py` (a 53-line subclass of `deepseek_v32.Model`); the actual MLA+DSA+IndexShare forward graph is in `mlx_lm/models/deepseek_v32.py`. **Not built or imported by the kitchen** — exists for code-reference only (e.g. the documented IndexShare blocker in `REAP37_EXPERIMENTS.md` / PLAN.md Issue 1.2). |
| `vendor/jangq` | `jjang-ai/jangq` main (`e70f220…`, depth 1) | JANGTQ TurboQuant runtime — `jang_tools.load_jangtq`, per-model JANGTQ converters, the MXTQ Metal matmul + decode kernels (`jang-runtime/Sources/JANGCoreMetal/JANGTQMatmul.metal`). Read-only upstream (not a fork); fork only if upstreaming a patch. Needed only for the JANGTQ_K MLX path. |
| `vendor/vmlx` | `jjang-ai/vmlx` main (`b7da1b8…`, v1.5.69, depth 1) | vMLX engine — `vmlx_engine.cli/server` + `utils/jang_loader.py` + `model_configs.py` (glm5 family + `cache=mla` registration). The server that serves a JANGTQ_K MLX bundle (`vmlx-serve serve ...`). Read-only upstream. Needed only for the JANGTQ_K MLX path. |

> Submodule state is preserved: `build_llamacpp.sh` detects when `$LLAMA_SRC`
> is the submodule (`.git` is a file, not a dir) and **skips the git
> fetch/checkout/reset steps** so the pinned feature-branch state stays intact.
> Only the cmake build runs against the submodule.

## Building patched llama.cpp binaries

```sh
bash mixed-precision-quantization/scripts/build_llamacpp.sh
```

Default behavior (no env vars set):
- `LLAMA_SRC` defaults to `$ROOT/vendor/llama.cpp` (the submodule)
- Detects submodule case → no git operations, just cmake build
- Builds `llama-cli`, `llama-quantize`, `llama-trace-moe`, `llama-gguf-split`
- Output: `$ROOT/vendor/llama.cpp/build-metal/bin/`
- `build-metal/` is included in llama.cpp's own `.gitignore` (`/build*`), so
  the submodule's git status stays clean after building.

Override `LLAMA_SRC="$HOME/projects/llama.cpp"` only if you want a separate
standalone clone (in which case the script DOES run the upstream
clone/reset-to-master path against it).

## Env vars read by operational scripts

These point at LARGE on-disk assets (model weights, REAP37 artifacts) that
don't live in any git repo. Binaries now default to the vendored submodule
build, but can be overridden when the build is elsewhere.

| Variable | Used by | Default if unset |
|---|---|---|
| `LLAMA_SRC` | `build_llamacpp.sh`, `quant_glm52_mixed.sh` | `$ROOT/vendor/llama.cpp` |
| `CLI` | `run_glm52_moe_trace.sh`, `baselines/*.sh` | `$ROOT/vendor/llama.cpp/build-metal/bin/llama-cli` (or `llama-trace-moe`) |
| `TRACE_BIN` | `run_trace_suite_batched.sh` | `$ROOT/vendor/llama.cpp/build-metal/bin/llama-trace-moe` |
| `MODEL` | `run_glm52_moe_trace.sh`, `baselines/*.sh` | Falls through `MODEL_DIR` → `/Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-mixed-…-00001-of-00009.gguf` |
| `MODEL_DIR` | `run_glm52_moe_trace.sh` (MODEL fallback step) | `/Volumes/Data NVME/GLM-5.2-GGUF` |
| `REAP37_MODEL_DIR` | `common/reap37/` scripts consuming the REAP37 MLX experimental model | none |

(`$ROOT` is resolved by each script as the kitchen checkout root, i.e. two
levels up from its own location in `common/scripts/` or
`mixed-precision-quantization/scripts/`.)

### `MODEL` resolution chain in `run_glm52_moe_trace.sh`

```text
MODEL="${MODEL:-${MODEL_DIR:-/Volumes/Data NVME/GLM-5.2-GGUF}/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/GLM-5.2-mixed-00001-of-00009.gguf}"
```

i.e. `MODEL` wins; else `MODEL_DIR` provides the parent dir; else a hard-coded
fallback path.

## Where these env vars are referenced

- `common/scripts/run_glm52_moe_trace.sh` reads `MODEL` (with `MODEL_DIR`
  fallback chain) and `CLI` (for the `llama-trace-moe` binary path).
- `common/baselines/glm52_*.sh` read `MODEL` and `CLI` similarly.
- `mixed-precision-quantization/scripts/build_llamacpp.sh` reads `LLAMA_SRC`
  to know where to build → `$LLAMA_SRC/build-metal/bin/llama-quantize`.
- `mixed-precision-quantization/scripts/quant_glm52_mixed.sh` reads
  `LLAMA_SRC` for the `llama-quantize` path.
- `AGENTS.md` references these env vars in operational-path lines.
- `GLM52_SESSION_MEMORY.md` and `GLM52_TRACE_PLAN.md` keep absolute paths in
  their narrative text because those are historical-factual records of the
  experiments; do not rewrite them as env vars.

## Quick start (zsh)

```sh
git clone <kitchen-url>
cd zai-glm-kitchen
git submodule update --init --recursive      # ~5 min for the llama.cpp clone

# Point at large on-disk GLM-5.2 assets (NOT in any repo):
export MODEL_DIR="/Volumes/Data NVME/GLM-5.2-GGUF"
export REAP37_MODEL_DIR="/Volumes/Data NVME/GLM-5.2-REAP37-MLX-4bit"

# Build patched llama.cpp binaries directly into the vendored submodule:
bash mixed-precision-quantization/scripts/build_llamacpp.sh

# Install the kitchen (auto-resolves gguf2mlx submodule via uv.sources):
uv pip install -e ".[dev]"

# Verify:
pytest -q                                       # 71 pass + 1 skip (scoped to tests/)
python -c "from gguf2mlx import convert; print('ok')"  # converter importable
```

## Run the model

Everything below assumes `MODEL` resolves to a built GGUF. Both baseline scripts
default it to the full mixed-IQ2S GGUF at
`$MODEL_DIR/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/GLM-5.2-mixed-00001-of-00009.gguf`;
override with `MODEL=...` to point at the shortgpt-pruned variant etc.

### Short context (merge-sort smoke)

```sh
bash common/baselines/glm52_merge_sort_baseline.sh
# writes merge-sort output + a 6-case Python sanity check to $OUT
# ENV:  MODEL  GGUF path (default: full mixed-IQ2S)
#       OUT    output file (default: glm52_merge_sort_output.txt)
#       CLI    llama-cli binary (default: $ROOT/vendor/llama.cpp/build-metal/bin/llama-cli)
```

### Long context (~18.7K-token needle-in-haystack retrieval)

```sh
bash common/baselines/glm52_longctx_retrieval_baseline.sh
# pre-token-counts the prompt, runs retrieval, checks for sentinel BLUE-FALCON-48217
# ENV:  MODEL        GGUF path (default: full mixed-IQ2S)
#       PROMPT_FILE needle prompt (default: common/baselines/long_coding_task_20k_retrieval_prompt.md)
#       OUT         output file (default: glm52_longctx_retrieval_output.txt)
# Expected answer: sentinel BLUE-FALCON-48217, function repair_event_stream, recursion_allowed: no
```

### Launch the server (for prompt-cache / decode benchmarks)

```sh
$ROOT/vendor/llama.cpp/build-metal/bin/llama-server \
  -m "$MODEL" --host 127.0.0.1 --port 8081 -ngl 999 -c 100000 -np 1 -fa on -t 8 --jinja --no-warmup
# then run e.g.:  .venv/bin/python scripts/longctx_decode_bench.py <label> <out_json>
```

### DSA sparse-gather attention (opt-in)

The decode-path sparse-gather rewrite (PLAN.md §7.N) is **off by default** — the
frozen dense baseline runs unless `LLAMA_DSA_SPARSE_GATHER=1` is set. Measured:
1.28× faster at 53K-context decode (4.93 vs 3.85 tok/s), 1.55× at short context,
with no correctness regression and a byte-identical prefill/cache path.

```sh
LLAMA_DSA_SPARSE_GATHER=1 bash common/baselines/glm52_merge_sort_baseline.sh  # short ctx, sparse
LLAMA_DSA_SPARSE_GATHER=1 vendor/llama.cpp/build-metal/bin/llama-server ...   # server, sparse
# unset (or omit the env var) to run the unchanged dense baseline
```

> ⚠️ The IQ2S-expert quantization tier collapses to incoherent output past ~8–16K
> context in **both** the full-mixed and shortgpt-pruned GGUFs (see
> `GLM52_SESSION_MEMORY.md`, 2026-06-24 §7.N entry). Short-context results remain
> correct; long-context *quality* cannot be validated at this quantization tier
> regardless of attention path.

## Quantize + run GLM-5.2 with JANGTQ_K (MLX path)

This is the **alternative to the GGUF/llama.cpp stack**: a non-uniform
TurboQuant (MXTQ codebook) MLX bundle served by vMLX. It stays **coherent and
correct at 53K context** (recovers the BLUE-FALCON-48217 needle) where the IQ2S
GGUFs collapse to gibberish — at the cost of lower tok/s and a larger bundle.

The kitchen's `mlx-export/convert_glm52_jangtq_k.py` is the **only** JANGTQ_K
path: `jang convert` / `vmlx-engine convert` expose only fixed `JANG_*` tiers,
not the per-projection mixed-bit TurboQuant profile.

### Prerequisites

- **vMLX installed** at `/Applications/vMLX.app` (bundled Python +
  `vmlx-serve` + `jang_tools`/`mlx` all live inside it).
- **BF16 HF source on disk**: `/Volumes/Backup/GLM-5.2` (~1.5 TB, 282 shards,
  hash-verified via `hf download zai-org/GLM-5.2`). A real fp16/BF16 source
  avoids the lossy-on-lossy GGUF→HF double-quantization that corrupted earlier
  JANG_4L experts.
- **~280 GB free** for the output bundle (JANGTQ_K is 279 GB / ~3.51 bpw).
- **vMLX bundled python** is mandatory for the converter so `jang_tools` /
  `mlx` are importable — the system `.venv` does **not** have them.

### 1. Quantize (BF16 → JANGTQ_K)

```sh
/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3 \
  mlx-export/convert_glm52_jangtq_k.py \
  /Volumes/Backup/GLM-5.2 \
  "/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-JANGTQ_K" \
  JANGTQ_K --clean
```

- Profile is optional (default `JANGTQ_K` = mixed). Uniform-bit alternatives:
  `JANGTQ2` / `JANGTQ3` / `JANGTQ4`.
- `--clean` removes orphan shards from a prior run (use for deterministic / CI
  runs; skip for resumable incremental conversion).
- Expected: ~2h wall, 277 shards, tq_bits histogram `{2: 38912, 4: 19456}`
  (zero 1-bit tensors — the JANG_4L crash cause is structurally absent).

Bit policy (JANGTQ_K mixed, per-projection on routed experts):

| projection | bits | why |
|---|---|---|
| `gate_proj`, `up_proj` | 2-bit MXTQ | gated activations — less sensitive |
| `down_proj` | 4-bit MXTQ | output enters the residual stream — most sensitive |
| `self_attn` (MLA/DSA), `shared_experts`, `embed_tokens`, `lm_head`, MoE router `gate`, all norms/biases | fp16 passthrough | keeping attention fp16 bypasses the `deepseek_v32.sanitize()` `bits=1` crash — see converter docstring |

### 2. Strip the MTP block (required, or the loader hard-fails)

GLM-5.2 ships `num_hidden_layers=78` (layers 0–77) **plus** a Multi-Token-
Prediction block at layer index 78 (`eh_proj`/`enorm`/`hnorm`/`shared_head`).
The `mlx_lm` `glm_moe_dsa` model class only instantiates layers 0–77, so
layer-78's routed-expert TQ groups have no target module and the JANGTQ
loader fails. MTP is a training/speculative-decode auxiliary; standard
autoregressive inference never uses it. Removing layer-78 tensors makes the
bundle load.

```sh
/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3 \
  mlx-export/strip_mtp_layer.py \
  "/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-JANGTQ_K"
```

Rewrites shards that mix layer-78 with real tensors (drops only the
`model.layers.78.*` keys), deletes shards that are purely layer-78, and
rewrites `model.safetensors.index.json`. Leaves `config.json`'s
`num_hidden_layers` untouched.

### 3. Serve + query (vMLX, OpenAI-compatible)

Requires a **one-time patch install** into the bundled vMLX (enables the MLA
prefix cache — without it, warm turns re-prefill the entire prefix every
request; see §"MLA prefix-cache patch" below):

```sh
# One-time: install the patched prefix_cache.py into the bundled vMLX
SITE=/Applications/vMLX.app/Contents/Resources/bundled-python/python/lib/python3.12/site-packages/vmlx_engine
cp "$(pwd)/vendor/vmlx/vmlx_engine/prefix_cache.py" "$SITE/prefix_cache.py"
# Re-run this after any vMLX app update (updates overwrite site-packages).
```

Then serve (note `--use-paged-cache` — required for the MLA prefix cache):

```sh
/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/vmlx-serve serve \
  "/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-JANGTQ_K" \
  --host 127.0.0.1 --port 8082 \
  --served-model-name glm52-jangtq-k \
  --max-num-seqs 1 --max-tokens 1024 --max-prompt-tokens 120000 \
  --timeout 7200 \
  --use-paged-cache
```

For development against the vendored source without installing, prefix the
command with
`PYTHONPATH="/Volumes/Data NVME/GLM-5.2-kitchen/vendor/vmlx"`.

Flags that matter:
- `--use-paged-cache` — **required** for MLA prefix-cache hits. The default
  memory-aware cache does full-key matching and misses the coding-agent
  pattern (long stable prefix + varied short question), so every warm turn
  re-prefills. The paged cache (block_size=64) matches shared prefixes with
  diverging suffixes and skips the cached prefill on warm turns.
- `--timeout 7200` (2h) — default 300s is too low for long-context prefill;
  a cold 53K-prompt request takes ~20+ min.
- `--max-prompt-tokens 120000` — needed for 53K-context tests (vMLX's preflight
  guard rejects otherwise).
- KV-cache quantization is auto-disabled for MLA models (MLA stores compressed
  latents that shouldn't be further quantized) — do not pass
  `--kv-cache-quantization`.

Quick query — **the `model` field is required** (vMLX returns HTTP 422
otherwise):

```sh
curl -s http://127.0.0.1:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm52-jangtq-k",
       "messages":[{"role":"user","content":"Write merge sort in Python."}],
       "max_tokens":512,"temperature":0.1}' | jq '.choices[0].message'
```

GLM-5.2 reasoning behavior: with thinking ON (default), the chain-of-thought
goes to `message.reasoning_content` and the final answer to `message.content`.
Short prompts can exhaust `max_tokens` inside reasoning before `content` is
written — either raise `max_tokens`, or pass
`chat_template_kwargs={"enable_thinking":false}` for direct answers.

### Coherence reference point

- **Short context**: coherent. Correct merge sort, 17×23=391, `is_prime`
  textbook wheel, multi-turn recall — ~9-10 tok/s gen.
- **53K long context** (BLUE-FALCON needle, thinking ON): **correct** — recovers
  `BLUE-FALCON-48217` from "Section 350" of the prompt. **Cold wall ~23.4 min**
  (prefill ~39 tok/s + decode ~6.9 tok/s); **warm wall ~60 s** (decode-only,
  prefix cache hit with `--use-paged-cache` + the patch) = **~24× speedup** on
  repeated coding-agent turns. This is the result that isolates the IQ2S GGUF
  gibberish as a quantization-tier failure, not a model / context-length /
  attention-path failure. See `GLM52_SESSION_MEMORY.md`, 2026-06-24 JANGTQ_K-vs-IQ2S 53K entry and the 2026-06-24 MLA prefix-cache-patch entry.

### MLA prefix-cache patch (vendor/vmlx)

The bundled vMLX `BlockAwarePrefixCache` had two bugs that together made the
MLA paged prefix cache unusable for GLM-5.2 (every warm turn re-prefilled):

1. **Store side** (`_extract_block_tensor_slice`, CacheList positional
   sub-cache branch): when a positional sub-cache had `seq_len==0` for a block
   (the legitimate MLA case where a layer's secondary head group accumulated
   zero tokens — live kshape `[.,.,0,feat]`), the store emitted `("skip",)`.
   Reconstruct then saw the sub as all-skip and aborted. **Fix:** store an
   empty zero-seq slice `[..., start:0, :]` instead, which preserves the
   feature dim and lets the standard `"kv"` rebuild path produce a real
   (non-None) empty KVCache matching the live state.
2. **Reconstruct side** (CacheList KV rebuild padding): MLA's compressed
   latent has **asymmetric k/v feature dims** (e.g. k ends in 64, v in 512).
   The padding code derived `pad_shape` from `ck.shape` and reused it for
   `v_pad`, crashing `mx.concatenate` when `offset % step != 0`. **Fix:**
   build pad shapes per-tensor (`k_pad_shape` from `ck.shape`, `v_pad_shape`
   from `cv.shape`).

The patch is in `vendor/vmlx/vmlx_engine/prefix_cache.py` (git submodule at
`b7da1b8` + the two-hunk diff). The bundled-site-packages copy is the runtime
path; reinstall it after any vMLX update. Both fixes are opt-in via
`--use-paged-cache` (default server runs are unaffected).

> ⚠️ Decode throughput (~6.9 tok/s) and cold prefill (~39 tok/s) are both lower
> than the GGUF/llama.cpp stack. But the MLA prefix-cache patch makes **warm
> turns decode-only** (~60 s at 53K), so repeated coding-agent turns
> (Architect/Reviewer with a stable long prefix) are fast regardless of the
> cold-start numbers. Remaining speed levers: vMLX `--prefill-batch-size`
> sweep (cold prefill only), MXTQ kernel OPT re-sweep, T>1 decode helpers.
> See `JANGTQ_K_SPEED_PLAN.md`.
