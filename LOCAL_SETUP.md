# Local setup

The kitchen is portable — no absolute paths are hardcoded in code or in the
run scripts. Three env vars are read at runtime when invoking operational
binaries; set them locally to point at your own filesystem layout.

## Variables

| Variable | Required by | Example |
|---|---|---|
| `LLAMA_CPP_DIR` | patched llama.cpp binaries (`llama-cli`, `llama-quantize`, `llama-trace-moe`) | `/Users/spotted/projects/llama.cpp` |
| `MODEL_DIR` | the mixed-precision GLM-5.2 GGUF (`$MODEL_DIR/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/...`) | `/Volumes/Data NVME/GLM-5.2-GGUF` |
| `REAP37_MODEL_DIR` | prebuilt REAP37 MLX model (experimental track) | `/Volumes/Data NVME/GLM-5.2-REAP37-MLX-4bit` |

## Where they're read

- `common/scripts/run_glm52_moe_trace.sh` reads `MODEL` env var with a
  default of `$MODEL_DIR/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/...`. Override
  `MODEL` to point at a different GGUF family (e.g. the ShortGPT-pruned
  variant) for trace runs against other model versions.
- `AGENTS.md` references these env vars in operational-path lines.
- `GLM52_SESSION_MEMORY.md` and `GLM52_TRACE_PLAN.md` keep absolute paths in
  their narrative text because those are historical-factual records of the
  experiments; do not rewrite them as env vars.

## Quick start (zsh)

```sh
export LLAMA_CPP_DIR="$HOME/projects/llama.cpp"
export MODEL_DIR="/Volumes/Data NVME/GLM-5.2-GGUF"
export REAP37_MODEL_DIR="/Volumes/Data NVME/GLM-5.2-REAP37-MLX-4bit"

# Install the kitchen as an editable package + run tests
uv pip install -e .
pytest -q
```
