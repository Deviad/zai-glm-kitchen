#!/usr/bin/env bash
# REAP37 MLX baseline 1: short coding task.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/Volumes/Data NVME/GLM-5.2-REAP37-MLX-4bit-indexer-compat}"
OUT="${OUT:-$ROOT/reap37_mlx_merge_sort_output.txt}"
PROMPT="${PROMPT:-Write down a merge sort algo non recursive in Python. Do not explain your reasoning. Output the Python code first, then one short sentence.}"
MAX_TOKENS="${MAX_TOKENS:-1400}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "FATAL: model not downloaded or config missing: $MODEL_DIR" >&2
  echo "Run: ./scripts/reap37/download_reap37_mlx.sh" >&2
  exit 1
fi

{
  echo "starting REAP37 MLX merge-sort baseline at $(date)"
  echo "model_dir=$MODEL_DIR"
  echo "prompt=$PROMPT"
  echo "max_tokens=$MAX_TOKENS"
} > "$OUT"

printf '%s' "$PROMPT" | /usr/bin/time -p uv run --with mlx-lm python -m mlx_lm generate \
  --model "$MODEL_DIR" \
  --prompt - \
  --max-tokens "$MAX_TOKENS" \
  --temp 0.1 \
  --top-p 0.95 \
  --min-p 0.01 \
  --chat-template-config '{"enable_thinking":false,"reasoning_effort":null}' \
  --verbose True >> "$OUT" 2>&1

STATUS=$?
echo "" >> "$OUT"
echo "exit_status=$STATUS at $(date)" >> "$OUT"

echo "Wrote: $OUT"
grep -E "Prompt:|Generation:|tokens-per-sec|tokens/sec|exit_status|sentinel:|function:|recursion_allowed:" "$OUT" | tail -20 || true
exit "$STATUS"
