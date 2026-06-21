#!/usr/bin/env bash
# Batched multilingual MoE routing trace.
#
# Loads the mixed GLM-5.2 model ONCE and traces N prompts from the expanded
# smoke suite in a single process (vs run_trace_task_suite.sh which reloads the
# model per prompt). ~4x faster for 2 prompts; the speedup grows with N because
# the ~27s model load is amortized across the whole batch.
#
# Each suite record is mapped to a PromptSpec {prompt, task_label=domain,
# language, script, prompt_family, test_id}. The C++ tracer writes one
# <test_id>-<language>.jsonl (+ .meta.json) per prompt to TRACE_OUT, then we run
# the analyzer to produce a multilingual routing report.
#
# Env:
#   SUITE        path to expanded smoke-suite JSONL
#   LANGS        space-separated subset (default = all 7)
#   DOMAINS      space-separated subset (default = all 7)
#   LIMIT        max number of suite records (0 = all)
#   N_PRED       max new tokens per prompt (default 16)
#   CTX          context size (default 4096)
#   NGL          GPU layers (default 999)
#   MODEL        path to GGUF (default = known-good mixed baseline)
#   TRACE_BIN    path to llama-trace-moe binary
#   PHASE        prefill|generation|both (default both)
#   TRACE_OUT    output dir (default traces/batch/<timestamp>)
#   REPORT_MD    report path (default reports/glm52_batched_trace_report.md)
#   SKIP_ANALYZE if 1, skip the analyzer step

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

SUITE="${SUITE:-prompts/tracing/glm52_trace_smoke_suite.expanded.jsonl}"
# NOTE: use `${VAR-default}` (no colon before dash) for LANGS/DOMAINS so empty
# string means "no filter, keep all" rather than falling back to defaults. With
# bash `${VAR:-default}`, empty `LANGS=""` collapses to the default 7-language
# list and silently filters out code-switch labels like `en+it` or `en+zh`.
# Set explicitly (e.g. `LANGS="en+it en+zh ..."`) to override both.
LANGS="${LANGS-en it zh es fr de pt}"
DOMAINS="${DOMAINS-coding math physics engineering computer_science chemistry cybersecurity}"
LIMIT="${LIMIT:-0}"
ONE_PER_COMBO="${ONE_PER_COMBO:-0}"
N_PRED="${N_PRED:-16}"
CTX="${CTX:-4096}"
NGL="${NGL:-999}"
MODEL="${MODEL:-/Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/GLM-5.2-mixed-00001-of-00009.gguf}"
TRACE_BIN="${TRACE_BIN:-$ROOT/vendor/llama.cpp/build-metal/bin/llama-trace-moe}"
PHASE="${PHASE:-both}"
TS="$(date +%Y%m%d-%H%M%S)"
TRACE_OUT="${TRACE_OUT:-traces/batch/$TS}"
REPORT_MD="${REPORT_MD:-reports/glm52_batched_trace_report.md}"
REPORT_JSON="${REPORT_JSON:-reports/glm52_batched_trace_summary.json}"

mkdir -p "$TRACE_OUT"
batch_prompts="$TRACE_OUT/.batch_prompts.jsonl"

# ---- filter the suite + map to PromptSpec schema ----------------------------
# Keep records matching LANGS/DOMAINS/LIMIT, then project to the PromptSpec the
# tracer consumes: {prompt, task_label=domain, language, script, prompt_family,
# test_id}. Default task_label/language to cfg fields if a record omits them.
python3 - "$SUITE" "$LANGS" "$DOMAINS" "$LIMIT" "$ONE_PER_COMBO" "$batch_prompts" <<'PY'
import json, sys
suite, langs_s, domains_s, limit_s, one_per, out_path = sys.argv[1:7]
langs = set(langs_s.split())
domains = set(domains_s.split())
limit = int(limit_s or 0)
# Treat "0"/""/"false"/"no"/"False" as falsy: a bare `if one_per` would be
# True for the non-empty string "0", silently forcing one-per-combo always.
_one_per_raw = (one_per or "").strip().lower()
one_per = _one_per_raw not in ("0", "", "false", "no", "off", "none")
seen_combos = set()
n = 0
with open(suite) as fh, open(out_path, "w") as out:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if langs and rec.get("language") not in langs:
            continue
        if domains and rec.get("domain") not in domains:
            continue
        if one_per:
            key = (rec.get("language"), rec.get("domain"))
            if key in seen_combos:
                continue
            seen_combos.add(key)
        spec = {
            "prompt":        rec["prompt"],
            "task_label":    rec.get("domain", "misc"),
            "language":      rec.get("language", "en"),
            "script":        rec.get("script", "Latin"),
            "prompt_family": rec.get("prompt_family", rec.get("domain", "misc")),
            "test_id":       rec.get("test_id", "adhoc"),
        }
        out.write(json.dumps(spec, ensure_ascii=False) + "\n")
        n += 1
        if limit and n >= limit:
            break
print(n)
PY

n_selected=$(grep -c . "$batch_prompts" 2>/dev/null || echo 0)
if [ "$n_selected" -eq 0 ]; then
  echo "ERROR: no suite records matched (suite=$SUITE langs=$LANGS domains=$DOMAINS)" >&2
  exit 2
fi
echo ">>> batched mode: $n_selected prompts -> model loaded once"

# ---- single batched C++ run -------------------------------------------------
echo ">>> bin: $TRACE_BIN"
echo ">>> model: $MODEL"
echo ">>> out dir: $TRACE_OUT"
echo ">>> ctx: $CTX  ngl: $NGL  n_pred: $N_PRED  phase: $PHASE"

# Build trace_args array (single batched C++ run). Activation tracing is
# opt-in: TRACE_ACTIVATIONS="l_out" adds --trace-activations + topk/stride.
trace_args=(
  -m "$MODEL"
  -ngl "$NGL" -c "$CTX" --temp 0.8 --predict "$N_PRED"
  --trace-prompts "$batch_prompts"
  --trace-out "$TRACE_OUT"
  --trace-phase "$PHASE" --trace-backpressure sample
)
if [ -n "${TRACE_LAYERS:-}" ]; then
  trace_args+=(--trace-layers "$TRACE_LAYERS")
fi
if [ -n "${TRACE_MAX_TOKENS:-}" ]; then
  trace_args+=(--trace-max-tokens "$TRACE_MAX_TOKENS")
fi
if [ -n "${TRACE_ACTIVATIONS:-}" ]; then
  trace_args+=(--trace-activations "$TRACE_ACTIVATIONS")
fi
if [ -n "${TRACE_ACTIVATION_TOPK:-}" ]; then
  trace_args+=(--trace-activation-topk "$TRACE_ACTIVATION_TOPK")
fi
if [ -n "${TRACE_ACTIVATION_STRIDE:-}" ]; then
  trace_args+=(--trace-activation-stride "$TRACE_ACTIVATION_STRIDE")
fi

"$TRACE_BIN" "${trace_args[@]}" \
  2>&1 | grep -iE "prompt tokens|trace written|batched|ERROR|failed to" \
  | tee "$TRACE_OUT/run.log" || true

echo ">>> done. traces in: $TRACE_OUT  (run log: $TRACE_OUT/run.log)"

# ---- analyze ----------------------------------------------------------------
if [ "${SKIP_ANALYZE:-0}" = "1" ]; then
  echo ">>> SKIP_ANALYZE=1: skipping analyzer"
  exit 0
fi

source "${ROOT}/.venv/bin/activate" 2>/dev/null || true
mkdir -p "$(dirname "$REPORT_MD")"
echo ">>> analyzing $TRACE_OUT/*.jsonl -> $REPORT_MD"

python3 common/scripts/analyze_moe_trace.py \
  --traces "$TRACE_OUT/*.jsonl" \
  --out-md "$REPORT_MD" \
  --out-json "$REPORT_JSON" \
  --topn 10 \
  2>&1 | tail -2

echo ">>> report: $REPORT_MD"
echo ">>> summary: $REPORT_JSON"
