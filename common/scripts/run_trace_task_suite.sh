#!/usr/bin/env bash
# Run the GLM-5.2 multilingual trace smoke suite under tracing and aggregate.
#
# Planned artifact from GLM52_TRACE_PLAN.md (Phase 1, Story 7):
#   common/scripts/run_trace_task_suite.sh
#
# Loops the expanded smoke suite (prompts/tracing/glm52_trace_smoke_suite.expanded.jsonl),
# invokes run_glm52_moe_trace.sh per prompt, then runs the analyzer to produce
# reports/glm52_moe_trace_report.md + summary JSON.
#
# Defaults to disabled thinking + small max_new_tokens for the smoke suite
# (per the trace plan's thinking-budget policy: smoke is for routing/activation
# comparability across languages/domains, not maximum reasoning quality).
#
# Env overrides (in addition to those honored by run_glm52_moe_trace.sh):
#   SUITE        path to expanded smoke-suite JSONL
#   TRACE_DIR    output dir for per-prompt traces (default traces/smoke/<ts>)
#   REPORT_DIR   output dir for report (default reports)
#   LIMIT        max number of suite records to run (0 = all)
#   LANGS        space-separated subset, e.g. "en zh" (default = all 7)
#   DOMAINS      space-separated subset, e.g. "coding math" (default = all)
#   SKIP_LIVE    if set to 1, skip the live C++ runs and only run the analyzer
#                on whatever trace files already exist in TRACE_DIR (use with synth)
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

SUITE="${SUITE:-prompts/tracing/glm52_trace_smoke_suite.expanded.jsonl}"
TS="$(date +%Y%m%d-%H%M%S)"
TRACE_DIR="${TRACE_DIR:-traces/smoke/$TS}"
REPORT_DIR="${REPORT_DIR:-reports}"
LIMIT="${LIMIT:-0}"
LANGS="${LANGS:-en it zh es fr de pt}"
DOMAINS="${DOMAINS:-coding math physics engineering computer_science chemistry cybersecurity}"
SKIP_LIVE="${SKIP_LIVE:-0}"

run_single="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_glm52_moe_trace.sh"

mkdir -p "$TRACE_DIR" "$REPORT_DIR"

# ---- filter the suite -------------------------------------------------------
# Read the JSONL with python (jq not assumed) and keep records matching the
# requested LANGS / DOMAINS subsets. Write to a temp file so the loop below is
# bash-3.2-safe (no mapfile/readarray, which is bash 4+).
selected_file="$TRACE_DIR/.selected.jsonl"
mkdir -p "$TRACE_DIR"
python3 - "$SUITE" "$LANGS" "$DOMAINS" "$LIMIT" "$selected_file" <<'PY'
import json, sys
suite, langs_s, domains_s, limit_s, out_path = sys.argv[1:6]
langs = set(langs_s.split())
domains = set(domains_s.split())
limit = int(limit_s or 0)
n = 0
with open(suite) as fh, open(out_path, "w") as out:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("language") not in langs:
            continue
        if rec.get("domain") not in domains:
            continue
        out.write(json.dumps(rec) + "\n")
        n += 1
        if limit and n >= limit:
            break
PY

n_selected=$(grep -c . "$selected_file" 2>/dev/null || echo 0)
if [ "$n_selected" -eq 0 ]; then
  echo "ERROR: no suite records matched (suite=$SUITE langs=$LANGS domains=$DOMAINS)" >&2
  exit 2
fi
echo ">>> selected $n_selected suite records"

# ---- live runs --------------------------------------------------------------
if [ "$SKIP_LIVE" = "1" ]; then
  echo ">>> SKIP_LIVE=1: skipping C++ runs; analyzing existing traces in $TRACE_DIR"
else
  n_done=0
  while IFS= read -r rec_json; do
    [ -z "$rec_json" ] && continue
    test_id="$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('test_id','adhoc'))" "$rec_json")"
    lang="$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('language','en'))" "$rec_json")"
    domain="$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('domain','misc'))" "$rec_json")"
    out="$TRACE_DIR/glm52-${domain}-${lang}-${test_id}.jsonl"
    if [ -e "$out" ]; then
      echo ">>> skip existing: $out"
      continue
    fi
    # write the prompt to a temp file so the C++ backend reads it via --file
    prompt_file="$TRACE_DIR/.prompts/${test_id}-${lang}.txt"
    mkdir -p "$TRACE_DIR/.prompts"
    python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('prompt',''))" "$rec_json" > "$prompt_file"
    echo ">>> [$((n_done+1))/$n_selected] $test_id ($lang, $domain) -> $out"
    if ! PROMPT_FILE="$prompt_file" \
         TASK_LABEL="$domain" \
         LANGUAGE="$lang" \
         SCRIPT="$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('script','Latin'))" "$rec_json")" \
         PROMPT_FAMILY="$domain" \
         TEST_ID="$test_id" \
         OUT="$out" \
         N_PRED="${N_PRED:-128}" \
         CTX="${CTX:-8192}" \
         bash "$run_single"; then
      echo "WARNING: trace failed for $test_id/$lang (continuing)" >&2
    fi
    n_done=$((n_done+1))
  done < "$selected_file"
  echo ">>> done $n_done live traces"
fi

# ---- analyze ----------------------------------------------------------------
python3 common/scripts/analyze_moe_trace.py \
  --traces "$TRACE_DIR/*.jsonl" \
  --out-md  "$REPORT_DIR/glm52_moe_trace_report.md" \
  --out-json "$REPORT_DIR/glm52_moe_trace_summary.json"

echo ">>> report: $REPORT_DIR/glm52_moe_trace_report.md"
echo ">>> summary: $REPORT_DIR/glm52_moe_trace_summary.json"
