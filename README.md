# GLM-5.2 research kitchen

Self-contained research bundle for GLM-5.2 work — the tracing framework, the
mixed-precision quantization scripts, and the layer-level structured pruning
artifacts that produced the ShortGPT-pruned spaced-N=12 model variant
(36 GB / 15.4% saved, all baselines passing).

Originally lived scattered inside the [`gguf2mlx`](https://github.com/barrontang/gguf2mlx)
converter repo (`scripts/`, `src/gguf2mlx/tracing/`, `reports/`, `traces/`,
`prompts/`). Extraction into this kitchen makes the work independently
portable + installable and lets the converter stay focused on GGUF→MLX
translation only.

## Vendored dependencies (git submodules)

| Path | Branch | Provides |
|---|---|---|
| `vendor/gguf2mlx` | `Deviad/gguf2mlx` @ `feature/update_for_glm5.2_cooking` | `from gguf2mlx import convert` — end-to-end GGUF→MLX translator, used for the Phase 8 MLX conversion step |
| `vendor/llama.cpp` | `Deviad/llama.cpp` @ `feature/patch_used_to_create_mixed_quantization_of_glm5.2` | patched `llama-cli`, `llama-quantize`, `llama-trace-moe` binaries; trace-moe example + BI computation needed by every trace run + the mixed-precision `quant_glm52_mixed.sh` |

Both are pinned to the tip of their respective fork feature branches; after a
fresh clone, run:

```sh
git submodule update --init --recursive      # ~5 min for the llama.cpp clone
```

## Layout

```
zai-glm-kitchen/
├── glm52_kitchen/                     # Python package (pip install -e .)
│   └── tracing/                       # MoE routing / activation / retrieval tracing
├── tests/                             # 4 tracing-related test modules
├── common/
│   ├── scripts/                       # 9 tracing CLI scripts (+ bash wrappers)
│   ├── baselines/                     # merge-sort + BLUE-FALCON retrieval baseline scripts
│   ├── prompts/                       # multilingual trace smoke suite (7 langs × 7 domains)
│   ├── reports/                       # 37 common-axes report artifacts
│   ├── traces/                        # common-axes trace datasets (batch/smoke/synth gitignored)
│   ├── phase3_dsa_unblock/            # rejected DSA-patch forensic artifacts
│   └── reap37/                        # REAP37 MLX 4-bit experimental track
├── mixed-precision-quantization/
│   └── scripts/                       # build_llamacpp.sh, quant_glm52_mixed.sh, verify_glm52_mixed.py
├── layer-level-structured-pruning/
│   ├── scripts/                       # prune_gguf.py, prune_layers.py, analyze_bi_scores.py
│   ├── reports/                       # 12 BI-plan artifacts + prune inventory
│   └── traces/                        # 327 trace files (BI calibration + forensic 3-way + baselines)
└── vendor/                            # git submodules
    ├── gguf2mlx/                      # converter fork (see above)
    └── llama.cpp/                     # patched llama.cpp fork (see above)
```

## Canonical docs (root)

- [`AGENTS.md`](AGENTS.md) — contract: how findings get tracked, what counts
  as a durable finding, where operational binaries live.
- [`LOCAL_SETUP.md`](LOCAL_SETUP.md) — submodule init + env vars
  (`LLAMA_CPP_DIR`, `MODEL_DIR`, `REAP37_MODEL_DIR`) + how to set them locally.
- [`PLAN.md`](PLAN.md) — GLM-5.2 `glm-dsa` converter feature plan (the
  converter implementation itself lives at `vendor/gguf2mlx`, see submodules).
- [`GLM52_SESSION_MEMORY.md`](GLM52_SESSION_MEMORY.md) — 3,751-line canonical
  findings narrative (Phase 1 → Phase 8 + post-mortem).
- [`GLM52_TRACE_PLAN.md`](GLM52_TRACE_PLAN.md) — tracing methodology +
  user stories / acceptance criteria.
- [`REAP37_EXPERIMENTS.md`](REAP37_EXPERIMENTS.md) — REAP37 MLX 4-bit track.

## Install

```sh
git submodule update --init --recursive      # first-time only
uv pip install -e .                           # registers glm52_kitchen + vendored gguf2mlx
pytest -q                                     # 71 tests pass + 1 skipped
python -c "from gguf2mlx import convert"      # converter available via submodule
```

The kitchen `pyproject.toml` declares the gguf2mlx submodule as a uv path
dependency, so `uv pip install -e .` resolves the converter automatically.

## Cross-references to outside

- gguf2mlx converter fork: https://github.com/Deviad/gguf2mlx/tree/feature/update_for_glm5.2_cooking (tracked as submodule at `vendor/gguf2mlx`)
- llama.cpp patched fork: https://github.com/Deviad/llama.cpp/tree/feature/patch_used_to_create_mixed_quantization_of_glm5.2 (tracked as submodule at `vendor/llama.cpp`)
- Both forks descend from the originals at https://github.com/barrontang/gguf2mlx and https://github.com/ggml-org/llama.cpp respectively.
