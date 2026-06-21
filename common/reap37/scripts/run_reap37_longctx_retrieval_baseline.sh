#!/usr/bin/env bash
# REAP37 MLX baseline 2: ~20k-token long-context retrieval.
# Expected answer:
#   sentinel: BLUE-FALCON-48217
#   function: repair_event_stream
#   recursion_allowed: no
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/Volumes/Data NVME/GLM-5.2-REAP37-MLX-4bit-indexer-compat}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT/long_coding_task_20k_retrieval_prompt.md}"
OUT="${OUT:-$ROOT/reap37_mlx_longctx_retrieval_output.txt}"
MAX_TOKENS="${MAX_TOKENS:-700}"
MAX_KV_SIZE="${MAX_KV_SIZE:-32768}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "FATAL: model not downloaded or config missing: $MODEL_DIR" >&2
  echo "Run: ./scripts/reap37/download_reap37_mlx.sh" >&2
  exit 1
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "FATAL: prompt missing: $PROMPT_FILE" >&2
  exit 1
fi

TOK_COUNT="unknown"
TOK="$HOME/projects/llama.cpp/build-metal/bin/llama-tokenize"
GGUF_MODEL="/Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/GLM-5.2-mixed-00001-of-00009.gguf"
if [[ -x "$TOK" && -f "$GGUF_MODEL" ]]; then
  TOK_COUNT="$($TOK --log-disable -m "$GGUF_MODEL" -f "$PROMPT_FILE" --show-count 2>/dev/null | awk '/Total number of tokens:/ {print $5}' | tail -1 || true)"
  [[ -n "$TOK_COUNT" ]] || TOK_COUNT="unknown"
fi

{
  echo "starting REAP37 MLX long-context retrieval baseline at $(date)"
  echo "model_dir=$MODEL_DIR"
  echo "prompt_file=$PROMPT_FILE"
  echo "prompt_tokens_estimate_with_glm_tokenizer=$TOK_COUNT"
  echo "max_kv_size=$MAX_KV_SIZE"
  echo "max_tokens=$MAX_TOKENS"
  echo "expected_sentinel=BLUE-FALCON-48217"
  echo "expected_function=repair_event_stream"
  echo "expected_recursion_allowed=no"
} > "$OUT"

cat "$PROMPT_FILE" | /usr/bin/time -p uv run --with mlx-lm python -m mlx_lm generate \
  --model "$MODEL_DIR" \
  --prompt - \
  --max-tokens "$MAX_TOKENS" \
  --max-kv-size "$MAX_KV_SIZE" \
  --temp 0.0 \
  --top-p 1.0 \
  --min-p 0.0 \
  --chat-template-config '{"enable_thinking":false,"reasoning_effort":null}' \
  --verbose True >> "$OUT" 2>&1

STATUS=$?
echo "" >> "$OUT"
echo "exit_status=$STATUS at $(date)" >> "$OUT"

echo "Wrote: $OUT"
echo "Token count estimate: $TOK_COUNT"
grep -E "sentinel:|function:|recursion_allowed:|Prompt:|Generation:|tokens-per-sec|tokens/sec|exit_status" "$OUT" | tail -30 || true
exit "$STATUS"
