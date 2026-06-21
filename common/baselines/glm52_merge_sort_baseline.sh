#!/usr/bin/env bash
# Baseline experiment 1: short coding task on the custom mixed GLM-5.2 GGUF.
# Expected previous result: coherent iterative merge sort; ~31.5 prompt tok/s,
# ~20.2 generation tok/s; generated code passed 6 simple Python sanity cases.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-/Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/GLM-5.2-mixed-00001-of-00009.gguf}"
CLI="${CLI:-$ROOT/vendor/llama.cpp/build-metal/bin/llama-cli}"
OUT="${OUT:-$ROOT/glm52_baseline_merge_sort_output.txt}"
PROMPT="${PROMPT:-Write down a merge sort algo non recursive in Python. Do not explain your reasoning. Output the Python code first, then one short sentence.}"

if [[ ! -x "$CLI" ]]; then
  echo "FATAL: llama-cli not found/executable: $CLI" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "FATAL: model shard not found: $MODEL" >&2
  exit 1
fi

{
  echo "starting merge-sort baseline at $(date)"
  echo "model=$MODEL"
  echo "prompt=$PROMPT"
} > "$OUT"

/usr/bin/time -p "$CLI" \
  -m "$MODEL" \
  -ngl 999 \
  -fa on \
  -c 4096 \
  -n 1400 \
  -t 16 \
  -tb 28 \
  --no-warmup \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":false,"reasoning_effort":null}' \
  -cnv -st \
  --temp 0.1 \
  --top-p 0.95 \
  --min-p 0.01 \
  -p "$PROMPT" >> "$OUT" 2>&1

STATUS=$?
echo "" >> "$OUT"
echo "exit_status=$STATUS at $(date)" >> "$OUT"

echo "Wrote: $OUT"
grep -E "Prompt:|Generation:|exit_status" "$OUT" | tail -10 || true
exit "$STATUS"
