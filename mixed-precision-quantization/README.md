# Mixed-precision quantization (GLM-5.2 IQ2_S experts + IQ4_NL rest)

The 232 GB / 2.64 BPW mixed-precision GGUF that serves as the canonical
baseline for every other experiment in this kitchen (ShortGPT pruning,
MLA retrieval studies, MoE routing analysis) is produced by these 3 scripts.

## Policy

| Tensor group | Quantization | Count |
|---|---|---|
| normal routed expert MLPs | **IQ2_S** | 225 |
| `blk.78` MTP routed experts | IQ4_NL (exception) | 3 |
| all non-experts (attention, norms, embeddings, output head) | IQ4_NL / F32 / Q6_K | — |

Rationale: experts dominate model size → IQ2_S for ~50% savings; everything
that touches routing / attention stays high-precision to preserve
reproducibility of trace data.

## Critical llama.cpp patch (issue #24379)

The quantizer used `n_layer()` (excludes the MTP layer) instead of
`n_layer_all` (includes `blk.78`) when sizing FFN/MoE tensor counts, so MTP
FFN tensors would silently miss the IQ2_S tier and stay at default
quantization. The patch lives at:

```text
$LLAMA_CPP_DIR/src/llama-quant.cpp
```

```diff
- qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)qs.model.hparams.n_layer();
+ qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)qs.model.hparams.n_layer_all;
```

Homebrew `llama-quantize` does NOT have this patch — must use the local
patched build (handle via `build_llamacpp.sh`).

## Scripts

| File | Purpose |
|---|---|
| `build_llamacpp.sh` | clones + builds the patched llama.cpp at `$LLAMA_CPP_DIR` |
| `quant_glm52_mixed.sh` | runs `llama-quantize` with the IQ2_S-experts + IQ4_NL-rest policy across all 9 shards. References `./build_llamacpp.sh` for the binary path resolution. |
| `verify_glm52_mixed.py` | cross-checks the output GGUF tensor → quantization mapping (`225 IQ2_S experts + 3 IQ4_NL MTP + 0 IQ2 non-experts` invariant) |

## Dry-run integration check

Run `quant_glm52_mixed.sh --dry-run` before a real quantization pass. It is
idempotent: it must not update `glm52_tensor_types.txt` or create GGUF output
shards. It validates the source shards, imatrix, generated tensor-type mapping,
GGUF metadata scan, size estimate, and then executes native
`llama-quantize --dry-run` with the same planned options to catch binary / CLI /
GLM-DSA / imatrix / tensor-type-file regressions. Python helper scans run via
`uv run --no-project --with gguf --with numpy python`. Exit code contract: `0` means all
checks passed; any non-zero status means dry-run failure or invalid CLI invocation
and stderr names the error.

## Model output location

The mixed-precision GGUF lands in `$MODEL_DIR/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/`
(9 shards, 232 GB). All downstream variants (`-pruned`, `-shortgpt`,
`-shortgpt-pruned`) are derived from this baseline via tensor-level operations
in `../layer-level-structured-pruning/scripts/`, never requantized.
