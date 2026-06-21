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

## Layout

```
GLM-5.2-kitchen/
├── glm52_kitchen/                     # Python package (pip install -e .)
│   └── tracing/                       # MoE routing / activation / retrieval tracing
├── tests/                             # 4 tracing-related test modules
├── common/
│   ├── scripts/                       # 9 tracing CLI scripts (+ bash wrappers)
│   ├── baselines/                     # merge-sort + BLUE-FALCON retrieval baseline scripts
│   ├── prompts/                       # multilingual trace smoke suite (7 langs × 7 domains)
│   ├── reports/                       # 37 common-axes report artifacts
│   ├── traces/                        # 3 common-axes trace datasets (batch/smoke/synth gitignored)
│   ├── phase3_dsa_unblock/            # rejected DSA-patch forensic artifacts
│   └── reap37/                        # REAP37 MLX 4-bit experimental track
├── mixed-precision-quantization/
│   └── scripts/                       # build_llamacpp.sh, quant_glm52_mixed.sh, verify_glm52_mixed.py
└── layer-level-structured-pruning/
    ├── scripts/                       # prune_gguf.py, prune_layers.py, analyze_bi_scores.py
    ├── reports/                       # 12 BI-plan artifacts + prune inventory
    └── traces/                        # 327 trace files (BI calibration + forensic 3-way + baselines)
```

## Canonical docs (root)

- [`AGENTS.md`](AGENTS.md) — contract: how findings get tracked, what counts
  as a durable finding, where operational binaries live.
- [`LOCAL_SETUP.md`](LOCAL_SETUP.md) — env vars (`LLAMA_CPP_DIR`,
  `MODEL_DIR`, `REAP37_MODEL_DIR`) + how to set them locally.
- [`PLAN.md`](PLAN.md) — GLM-5.2 `glm-dsa` converter feature plan (the
  converter implementation itself lives at github.com/barrontang/gguf2mlx).
- [`GLM52_SESSION_MEMORY.md`](GLM52_SESSION_MEMORY.md) — 3,751-line canonical
  findings narrative (Phase 1 → Phase 8 + post-mortem).
- [`GLM52_TRACE_PLAN.md`](GLM52_TRACE_PLAN.md) — tracing methodology +
  user stories / acceptance criteria.
- [`REAP37_EXPERIMENTS.md`](REAP37_EXPERIMENTS.md) — REAP37 MLX 4-bit track.

## Install

```sh
uv pip install -e .          # registers the `glm52_kitchen` package
pytest -q                    # 71 tests pass + 1 skipped
```

## Cross-references to outside

- gguf2mlx converter: https://github.com/barrontang/gguf2mlx
- llama.cpp patch site: see `AGENTS.md` → "Patched llama.cpp"
- Patched llama.cpp fork (issue #24379 + trace-moe example): built via
  `mixed-precision-quantization/scripts/build_llamacpp.sh`.
