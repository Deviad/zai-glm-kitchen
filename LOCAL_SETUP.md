# Local setup

The kitchen is portable — no code paths are hardcoded. Two external
dependencies are vendored as git submodules inside `vendor/`; everything else
is owned by the kitchen itself.

## Vendored submodules

After a fresh clone, init submodules once:

```sh
git clone <kitchen-url>
cd GLM-5.2-kitchen
git submodule update --init --recursive      # ~5 min for the llama.cpp clone
```

| Submodule | Branch (Deviad fork) | Provides |
|---|---|---|
| `vendor/gguf2mlx` | `feature/update_for_glm5.2_cooking` (`90affca2…`) | Imported at runtime as `from gguf2mlx import convert` (declared as a uv path dep in `pyproject.toml`). End-to-end GGUF→MLX translator for the Phase 8 MLX conversion step. |
| `vendor/llama.cpp` | `feature/patch_used_to_create_mixed_quantization_of_glm5.2` (`6f67b8a…`) | Pinned source of the patched llama.cpp fork. The patched build produces `llama-cli`, `llama-quantize`, `llama-trace-moe`, `llama-tokenize`, `llama-gguf-split` from `$ROOT/vendor/llama.cpp/build-metal/bin/`. The trace-moe example + ShortGPT Block Influence computation live in `examples/trace-moe/` inside the submodule. |

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
cd GLM-5.2-kitchen
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
