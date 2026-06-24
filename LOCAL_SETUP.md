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
