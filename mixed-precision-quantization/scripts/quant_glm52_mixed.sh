#!/usr/bin/env bash
# Quantize GLM-5.2 to a custom mixed precision:
#   routed-expert MLP weights -> 2-bit (IQ2_S)
#   everything else           -> 4-bit (IQ4_NL base, preserved)
#
# Source: the existing Unsloth UD-IQ4_NL (9 shards, 347 GB).
# Output: new sharded GGUF (~9 shards, est ~240-260 GB).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

# Default to the vendored llama.cpp submodule. Override LLAMA_SRC only if you
# want to use a separate build (in which case run build_llamacpp.sh against it).
LLAMA_SRC="${LLAMA_SRC:-$ROOT/vendor/llama.cpp}"
LQ="$LLAMA_SRC/build-metal/bin/llama-quantize"

IN_DIR="/Volumes/Data NVME/GLM-5.2-GGUF/UD-IQ4_NL"
IN_SHARD="$IN_DIR/GLM-5.2-UD-IQ4_NL-00001-of-00009.gguf"

# Unsloth's importance matrix (used to create the UD-IQ4_NL source). Required
# for IQ2_* quant types. Downloaded from unsloth/GLM-5.2-GGUF repo root.
IMATRIX="/Volumes/Data NVME/GLM-5.2-GGUF/imatrix_unsloth.gguf"

OUT_DIR="/Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest"
OUT_PREFIX="$OUT_DIR/GLM-5.2-mixed"
TENSOR_TYPES="/Volumes/Data NVME/GLM-5.2-GGUF/glm52_tensor_types.txt"

# --- 2-bit variant: override here if you want a different one ---
#   IQ2_S   (2.56 bpw)  <-- default, robust without imatrix
#   IQ2_M   (2.66 bpw)  best 2-bit quality
#   IQ2_XS  (2.31 bpw)  smaller; benefits from imatrix
#   IQ2_XXS (2.06 bpw)  smallest; benefits from imatrix
TWO_BIT="${TWO_BIT:-IQ2_S}"
BASE_BIT="${BASE_BIT:-IQ4_NL}"
NTHREADS="${NTHREADS:-28}"

if [[ ! -x "$LQ" ]]; then
  echo "FATAL: $LQ not found. Run ./build_llamacpp.sh first." >&2
  exit 1
fi
if [[ ! -f "$IN_SHARD" ]]; then
  echo "FATAL: input shard not found: $IN_SHARD" >&2
  exit 1
fi

# Regenerate the tensor-type file with the chosen 2-bit variant.
# IMPORTANT: llama.cpp applies tensor-type rules as regex_search() and FIRST
# MATCH WINS. Unsloth's imatrix has no entries for the MTP/NextN expert tensors
# in blk.78, so keep those at the high-precision base type and put those more
# specific rules before the generic routed-expert rules.
cat > "$TENSOR_TYPES" <<EOF
blk\\.78\\.ffn_down_exps=$BASE_BIT
blk\\.78\\.ffn_gate_exps=$BASE_BIT
blk\\.78\\.ffn_up_exps=$BASE_BIT
ffn_gate_exps=$TWO_BIT
ffn_up_exps=$TWO_BIT
ffn_down_exps=$TWO_BIT
EOF

mkdir -p "$OUT_DIR"

echo "==> source : $IN_DIR  (9 shards, IQ4_NL)"
echo "==> output : $OUT_DIR"
if [[ ! -f "$IMATRIX" ]]; then
  echo "FATAL: imatrix not found: $IMATRIX" >&2
  echo "  download: curl -L -o '$IMATRIX' https://huggingface.co/unsloth/GLM-5.2-GGUF/resolve/main/imatrix_unsloth.gguf_file" >&2
  exit 1
fi

echo "==> mapping: experts(3 fragments) -> $TWO_BIT | rest -> $BASE_BIT"
echo "==> imatrix: $IMATRIX"
echo "==> threads: $NTHREADS   keep-split: yes   allow-requantize: yes"
echo ""

time "$LQ" \
  --allow-requantize \
  --keep-split \
  --imatrix "$IMATRIX" \
  --tensor-type-file "$TENSOR_TYPES" \
  "$IN_SHARD" \
  "$OUT_PREFIX.gguf" \
  "$BASE_BIT" \
  "$NTHREADS"

echo ""
echo "==> DONE. Output shards:"
ls -lh "$OUT_DIR"/*.gguf
du -sh "$OUT_DIR"
