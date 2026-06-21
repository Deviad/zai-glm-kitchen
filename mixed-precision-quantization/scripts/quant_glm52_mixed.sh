#!/usr/bin/env bash
# Quantize GLM-5.2 to a custom mixed precision:
#   routed-expert MLP weights -> 2-bit (IQ2_S)
#   everything else           -> 4-bit (IQ4_NL base, preserved)
#
# Source: the existing Unsloth UD-IQ4_NL (9 shards, 347 GB).
# Output: new sharded GGUF (~9 shards, est ~240-260 GB).
set -euo pipefail

# --- Flags ---
DRY_RUN=0
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --dry-run|-n) DRY_RUN=1; shift ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

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

emit_tensor_types() {
  cat <<EOF
blk\\.78\\.ffn_down_exps=$BASE_BIT
blk\\.78\\.ffn_gate_exps=$BASE_BIT
blk\\.78\\.ffn_up_exps=$BASE_BIT
ffn_gate_exps=$TWO_BIT
ffn_up_exps=$TWO_BIT
ffn_down_exps=$TWO_BIT
EOF
}

if [[ ! -x "$LQ" ]]; then
  echo "FATAL: $LQ not found. Run ./build_llamacpp.sh first." >&2
  exit 1
fi
if [[ ! -f "$IN_SHARD" ]]; then
  echo "FATAL: input shard not found: $IN_SHARD" >&2
  exit 1
fi

# --- Dry-run: parse, validate, scan, estimate, print command, exit ---
if [[ $DRY_RUN -eq 1 ]]; then
  echo "============================================"
  echo "  Mixed-precision quantization -- DRY RUN"
  echo "============================================"
  echo ""

  echo "--- Source material ---"
  PASS=0
  ERRORS=0
  if [[ -f "$IN_SHARD" ]]; then
    echo "[OK] Source input shard exists"
    PASS=$((PASS+1))
  else
    echo "[FAIL] Source input shard missing: $IN_SHARD"
    ERRORS=$((ERRORS+1))
  fi
  NSHARDS=$(find "$IN_DIR" -maxdepth 1 -type f -name '*.gguf' | wc -l | tr -d ' ')
  if [[ "$NSHARDS" -eq 9 ]]; then
    echo "[OK] Source directory has 9 shards ($NSHARDS)"
    PASS=$((PASS+1))
  else
    echo "[FAIL] Expected 9 shards in $IN_DIR, found $NSHARDS"
    ERRORS=$((ERRORS+1))
  fi
  if [[ -f "$IMATRIX" ]]; then
    echo "[OK] Importance matrix exists"
    PASS=$((PASS+1))
  else
    echo "[FAIL] Importance matrix not found: $IMATRIX"
    ERRORS=$((ERRORS+1))
  fi
  echo "  Checks passed: $PASS"
  if [[ "$ERRORS" -ne 0 ]]; then
    echo "ERROR: dry-run validation failed with $ERRORS error(s)." >&2
    exit 1
  fi
  echo ""

  echo "--- Tensor-type mapping (would be generated at: $TENSOR_TYPES) ---"
  emit_tensor_types
  echo ""

  # Source model scan via uv run (heredoc to avoid shell-quoting issues)
  echo "--- Source metadata + tensor breakdown (shard 2) ---"
  export PY_IN_DIR="$IN_DIR" PY_BASE_BIT="$BASE_BIT" PY_TWO_BIT="$TWO_BIT"
  if ! uv run --with gguf --with numpy python << 'PYEOF'
import gguf, os
r_dir = os.environ.get('PY_IN_DIR', '')
r1 = gguf.GGUFReader(f'{r_dir}/GLM-5.2-UD-IQ4_NL-00001-of-00009.gguf')
for k in ['general.architecture', 'general.file_type',
          'glm-dsa.block_count', 'glm-dsa.embedding_length', 'glm-dsa.expert_count']:
    f = r1.get_field(k)
    v = f.contents() if f else 'N/A'
    if hasattr(v, 'decode'): v = v.decode().strip()
    elif hasattr(v, 'tolist'): v = v.tolist()[0]
    print(f'  {k}: {v}')
print()
r2 = gguf.GGUFReader(f'{r_dir}/GLM-5.2-UD-IQ4_NL-00002-of-00009.gguf')
ef = ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')
nt = len(r2.tensors)
ne = sum(1 for t in r2.tensors if any(f in t.name for f in ef))
nm = sum(1 for t in r2.tensors if any(f in t.name for f in ef) and 'blk.78.' in t.name)
bb = os.environ.get('PY_BASE_BIT', 'IQ4_NL')
tb = os.environ.get('PY_TWO_BIT', 'IQ2_S')
print(f'  Total tensors (shard 2) : {nt}')
print(f'  Expert tensors          : {ne}')
print(f'    -> blk.78 MTP         : {nm} (kept at {bb})')
print(f'    -> Normal routed       : {ne - nm} (-> {tb})')
print(f"  Non-expert tensors      : {nt - ne}")
print()
tt = {}
for t in r2.tensors:
    try: tn = gguf.GGMLQuantizationType(t.tensor_type).name
    except: tn = f'type={int(t.tensor_type)}'
    tt[tn] = tt.get(tn, 0) + 1
for k2, v2 in sorted(tt.items(), key=lambda x: -x[1]):
    print(f'    {v2:>5d}  {k2}')
PYEOF
  then
    echo "FATAL: GGUF source scan failed" >&2
    exit 1
  fi
  unset PY_IN_DIR PY_BASE_BIT PY_TWO_BIT

  # Size estimate
  echo ""
  echo "--- Estimated output (total source size × empirical mixed ratio) ---"
  export PY_IN_DIR="$IN_DIR"
  if ! uv run --with gguf --with numpy python << 'PYEOF'
import glob
import os

r_dir = os.environ.get("PY_IN_DIR", "")
shards = sorted(glob.glob(os.path.join(r_dir, "*.gguf")))
if not shards:
    raise SystemExit(f"no GGUF shards found under {r_dir!r}")

total = sum(os.path.getsize(p) for p in shards)
# From the verified quantization log saved in GLM52_SESSION_MEMORY.md:
#   source model size: 355388.74 MiB, quant size: 237634.13 MiB
ratio = 237634.13 / 355388.74
est = total * ratio
print(f"  Source shards scanned : {len(shards)}")
print(f"  Source total          : {total / 1e9:.1f} GB / {total / (1024**3):.1f} GiB")
print(f"  Empirical ratio       : {ratio:.4f} (237634.13 / 355388.74 MiB)")
print(f"  Est. mixed output     : ~{est / 1e9:.1f} GB / ~{est / (1024**3):.1f} GiB")
print("  Verified prior output : 237634.13 MiB / 2.64 BPW (~232 GiB)")
PYEOF
  then
    echo "ERROR: dry-run size estimation failed." >&2
    exit 1
  fi
  unset PY_IN_DIR

  # Full command preview
  echo ""
  echo "--- Full quantize command (NOT executed) ---"
  echo "  $LQ \\"
  echo "    --allow-requantize \\"
  echo "    --keep-split \\"
  echo "    --imatrix '$IMATRIX' \\"
  echo "    --tensor-type-file '$TENSOR_TYPES' \\"
  echo "    '$IN_SHARD' \\"
  echo "    '$OUT_PREFIX.gguf' \\"
  echo "    '$BASE_BIT' \\"
  echo "    '$NTHREADS'"

  echo ""
  echo "--- Native llama-quantize --dry-run integration check (executed) ---"
  echo "  Validates binary, GLM-DSA loading, imatrix, tensor-type parsing, keep-split, and quant-size calculation."
  if ! "$LQ" \
    --dry-run \
    --allow-requantize \
    --keep-split \
    --imatrix "$IMATRIX" \
    --tensor-type-file <(emit_tensor_types) \
    "$IN_SHARD" \
    "$OUT_PREFIX.gguf" \
    "$BASE_BIT" \
    "$NTHREADS"; then
    echo "ERROR: native llama-quantize --dry-run integration check failed." >&2
    exit 1
  fi

  echo ""
  echo "--- Summary ---"
  echo "  Status: DRY RUN COMPLETE — no files written, no quantization"
  echo "  Next step: run quant_glm52_mixed.sh without --dry-run to execute"
  echo "============================================"
  exit 0
fi

# Regenerate the tensor-type file with the chosen 2-bit variant.
# IMPORTANT: llama.cpp applies tensor-type rules as regex_search() and FIRST
# MATCH WINS. Unsloth's imatrix has no entries for the MTP/NextN expert tensors
# in blk.78, so keep those at the high-precision base type and put those more
# specific rules before the generic routed-expert rules.
emit_tensor_types > "$TENSOR_TYPES"

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
