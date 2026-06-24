#!/usr/bin/env bash
# Baseline experiment 2: ~20k-token long-context retrieval on the custom mixed
# GLM-5.2 GGUF.
# Expected answer:
#   sentinel: BLUE-FALCON-48217
#   function: repair_event_stream
#   recursion_allowed: no
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-/Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/GLM-5.2-mixed-00001-of-00009.gguf}"
CLI="${CLI:-$ROOT/vendor/llama.cpp/build-metal/bin/llama-cli}"
TOK="${TOK:-$ROOT/vendor/llama.cpp/build-metal/bin/llama-tokenize}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT/common/baselines/long_coding_task_20k_retrieval_prompt.md}"
OUT="${OUT:-$ROOT/glm52_baseline_longctx_retrieval_output.txt}"

if [[ ! -x "$CLI" ]]; then
  echo "FATAL: llama-cli not found/executable: $CLI" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "FATAL: model shard not found: $MODEL" >&2
  exit 1
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "FATAL: prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

TOK_COUNT="unknown"
if [[ -x "$TOK" ]]; then
  TOK_COUNT="$($TOK --log-disable -m "$MODEL" -f "$PROMPT_FILE" --show-count 2>/dev/null | awk '/Total number of tokens:/ {print $5}' | tail -1 || true)"
  [[ -n "$TOK_COUNT" ]] || TOK_COUNT="unknown"
fi

{
  echo "starting long-context retrieval baseline at $(date)"
  echo "model=$MODEL"
  echo "prompt_file=$PROMPT_FILE"
  echo "prompt_tokens_exact_pre_template=$TOK_COUNT"
  echo "expected_sentinel=BLUE-FALCON-48217"
  echo "expected_function=repair_event_stream"
  echo "expected_recursion_allowed=no"
} > "$OUT"

/usr/bin/time -p "$CLI" \
  -m "$MODEL" \
  -ngl 999 \
  -fa on \
  -c 32768 \
  -n 700 \
  -t 16 \
  -tb 28 \
  --no-warmup \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":false,"reasoning_effort":null}' \
  -cnv -st \
  --temp 0.0 \
  --top-p 1.0 \
  --min-p 0.0 \
  -f "$PROMPT_FILE" >> "$OUT" 2>&1

STATUS=$?
echo "" >> "$OUT"
echo "exit_status=$STATUS at $(date)" >> "$OUT"

echo "Wrote: $OUT"
echo "Token count pre-template: $TOK_COUNT"
grep -E "sentinel:|function:|recursion_allowed:|Prompt:|Generation:|exit_status" "$OUT" | tail -20 || true
exit "$STATUS"
