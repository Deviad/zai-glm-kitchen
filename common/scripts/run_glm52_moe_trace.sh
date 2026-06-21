#!/usr/bin/env bash
# Run a single traced GLM-5.2 MoE prompt and emit trace JSONL + metadata.
#
# Planned artifact from GLM52_TRACE_PLAN.md (Phase 1, Story 1):
#   common/scripts/run_glm52_moe_trace.sh
#
# This wraps the C++ trace-moe backend at:
#   /Users/spotted/projects/llama.cpp/build-metal/bin/llama-trace-moe
#   (legacy path; default now resolves to $ROOT/vendor/llama.cpp submodule)
#
# Defaults to the known-good mixed GGUF baseline. Honors env overrides:
#   MODEL       /path/to/model.gguf
#   CLI         /path/to/llama-trace-moe
#   OUT         /path/to/trace.jsonl
#   PROMPT_FILE /path/to/prompt.txt            (or inline PROMPT_TEXT)
#   TASK_LABEL  coding|math|physics|...
#   LANGUAGE    en|it|zh|es|fr|de|pt
#   SCRIPT      Latin|Han|mixed
#   PROMPT_FAMILY coding|science-reading|explanation|...
#   TEST_ID     coding_01_iterative_merge_sort
#   NGL         number of GPU layers (default 999)
#   CTX         context size (default 32768)
#   N_PRED      max new tokens (default 256)
#   PHASE       prefill|generation|both (default both)
#   TRACE_LAYERS  e.g. "0,1,2,6,10,42" or "0..78" (default all)
#   TRACE_MAX_TOKENS  cap traced tokens (default unlimited)
#   BACKPRESSURE  block|drop|sample (default sample)
#
# Exit codes: 0 success, 2 missing deps, 3 trace-moe failed, 4 exists already.
set -euo pipefail

# Resolve kitchen root (this script lives in common/scripts/) so default
# paths can refer to the vendored llama.cpp submodule at vendor/.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

# ---- defaults ---------------------------------------------------------------
MODEL="${MODEL:-${MODEL_DIR:-/Volumes/Data NVME/GLM-5.2-GGUF}/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/GLM-5.2-mixed-00001-of-00009.gguf}"
# Default trace-moe binary lives in the patched llama.cpp built from the
# vendor/llama.cpp submodule. Override CLI if your build is elsewhere.
CLI="${CLI:-$ROOT/vendor/llama.cpp/build-metal/bin/llama-trace-moe}"
TS="$(date +%Y%m%d-%H%M%S)"
TASK_LABEL="${TASK_LABEL:-coding}"
LANGUAGE="${LANGUAGE:-en}"
SCRIPT_TAG="${SCRIPT:-Latin}"
PROMPT_FAMILY="${PROMPT_FAMILY:-coding}"
TEST_ID="${TEST_ID:-adhoc}"
OUT="${OUT:-traces/glm52-${TASK_LABEL}-${LANGUAGE}-${TEST_ID}-${TS}.jsonl}"
NGL="${NGL:-999}"
CTX="${CTX:-32768}"
N_PRED="${N_PRED:-256}"
PHASE="${PHASE:-both}"
TRACE_LAYERS="${TRACE_LAYERS:-}"
TRACE_MAX_TOKENS="${TRACE_MAX_TOKENS:-}"
BACKPRESSURE="${BACKPRESSURE:-sample}"

# ---- preflight --------------------------------------------------------------
if [ ! -x "$CLI" ]; then
  echo "ERROR: trace-moe binary not found/not executable: $CLI" >&2
  echo "  build it from the vendored llama.cpp submodule at $ROOT/vendor/llama.cpp:" >&2
  echo "    bash mixed-precision-quantization/scripts/build_llamacpp.sh" >&2
  exit 2
fi
if [ ! -f "$MODEL" ]; then
  echo "ERROR: model not found: $MODEL" >&2
  exit 2
fi

# ---- prompt handling --------------------------------------------------------
prompt_file=""
prompt_text=""
if [ -n "${PROMPT_FILE:-}" ] && [ -f "$PROMPT_FILE" ]; then
  prompt_file="$PROMPT_FILE"
elif [ -n "${PROMPT_TEXT:-}" ]; then
  prompt_text="$PROMPT_TEXT"
else
  echo "ERROR: provide PROMPT_FILE or PROMPT_TEXT" >&2
  exit 2
fi

if [ -e "$OUT" ]; then
  echo "ERROR: trace output already exists (refusing to overwrite): $OUT" >&2
  exit 4
fi
mkdir -p "$(dirname "$OUT")"

# ---- command line ----------------------------------------------------------
trace_args=(
  --model "$MODEL"
  -ngl "$NGL"
  --ctx-size "$CTX"
  --predict "$N_PRED"
  --temp 0.0
  --jinja -cnv -st
  --chat-template-kwargs '{"enable_thinking":false,"reasoning_effort":null}'
  --trace-out "$OUT"
  --trace-task-label "$TASK_LABEL"
  --trace-language "$LANGUAGE"
  --trace-script "$SCRIPT_TAG"
  --trace-prompt-family "$PROMPT_FAMILY"
  --trace-test-id "$TEST_ID"
  --trace-phase "$PHASE"
  --trace-backpressure "$BACKPRESSURE"
)
# Allow caller to override prefill/decode batch sizes. Long-context runs (e.g.
# the BLUE-FALCON-48217 retrieval prompt at 18,745 tokens) need n_batch high
# enough to swallow the whole prompt in one prefill chunk — default would
# hit GGML_ASSERT(n_tokens_all <= cparams.n_batch).
if [ -n "${TRACE_BATCH_SIZE:-}" ]; then
  trace_args+=(--batch-size "${TRACE_BATCH_SIZE}")
fi
if [ -n "$prompt_file" ]; then
  trace_args+=(--file "$prompt_file")
elif [ -n "$prompt_text" ]; then
  trace_args+=(--prompt "$prompt_text")
fi
if [ -n "$TRACE_LAYERS" ]; then
  trace_args+=(--trace-layers "$TRACE_LAYERS")
fi
if [ -n "$TRACE_MAX_TOKENS" ]; then
  trace_args+=(--trace-max-tokens "$TRACE_MAX_TOKENS")
fi
# Story 6 AC: bounded activation summaries. Pass through only when set,
# so existing callers see no change. Empty/absent = off (full activation
# dumps are disabled by default per AC 6.5).
if [ -n "${TRACE_ACTIVATIONS:-}" ]; then
  trace_args+=(--trace-activations "$TRACE_ACTIVATIONS")
fi
if [ -n "${TRACE_ACTIVATION_TOPK:-}" ]; then
  trace_args+=(--trace-activation-topk "$TRACE_ACTIVATION_TOPK")
fi
if [ -n "${TRACE_ACTIVATION_STRIDE:-}" ]; then
  trace_args+=(--trace-activation-stride "$TRACE_ACTIVATION_STRIDE")
fi

# stash env context for the metadata sidecar (the C++ tool writes .meta.json;
# we record provenance echo here so it's in the run log too).
run_log="${OUT%.jsonl}.run.log"
{
  echo "# GLM-5.2 MoE trace run"
  echo "run_id: $TEST_ID-$LANGUAGE-$TS"
  echo "model: $MODEL"
  echo "cli: $CLI"
  echo "task_label: $TASK_LABEL"
  echo "language: $LANGUAGE"
  echo "script: $SCRIPT_TAG"
  echo "prompt_family: $PROMPT_FAMILY"
  echo "test_id: $TEST_ID"
  echo "ctx: $CTX  ngl: $NGL  n_pred: $N_PRED  phase: $PHASE"
  echo "backpressure: $BACKPRESSURE"
  if [ -n "$prompt_file" ]; then
    echo "prompt_file: $prompt_file"
    sha256sum "$prompt_file" 2>/dev/null || shasum -a 256 "$prompt_file"
  fi
  echo "command: $CLI ${trace_args[*]}"
  echo "----"
} > "$run_log"

echo ">>> writing trace -> $OUT"
if ! "$CLI" "${trace_args[@]}" 2>&1 | tee -a "$run_log"; then
  echo "ERROR: trace-moe failed (see $run_log)" >&2
  exit 3
fi

# ---- post-run ---------------------------------------------------------------
# The C++ backend writes <OUT>.meta.json. If it didn't (older build), nothing
# to fix up here — the analyzer tolerates a missing sidecar.
if [ -f "${OUT}.meta.json" ]; then
  echo ">>> metadata: ${OUT}.meta.json"
else
  echo ">>> WARNING: no ${OUT}.meta.json written — analyzer will run without run metadata" >&2
fi
echo ">>> done. trace: $OUT  log: $run_log"
