# `common/` — shared tracing infrastructure + reports + baselines + prompts

Holds the tracing framework's *data* layer: CLI scripts, prompt suites,
baseline scripts, report artifacts, and the common-axes trace datasets. Also
hosts the rejected DSA-patch forensic record and the REAP37 experimental
track.

The Python package itself lives at `../glm52_kitchen/tracing/` and is what
gets imported as `glm52_kitchen.tracing.*`. The scripts here are thin CLI
wrappers that consume trace JSONL produced by the C++ `llama-trace-moe`
binary in the patched llama.cpp tree.

## Layout

| Subpath | Contents |
|---|---|
| `scripts/` | 9 tracing CLIs + 3 bash wrappers (`analyze_moe_trace.py`, `compare_trace_reports.py`, `make_synth_trace.py`, `expand_smoke_suite.py`, `analyze_activation_cross_task.py`, `analyze_channel_focus.py`, `run_glm52_moe_trace.sh`, `run_trace_task_suite.sh`, `run_trace_suite_batched.sh`) |
| `baselines/` | `glm52_merge_sort_baseline.sh` + `glm52_longctx_retrieval_baseline.sh` + the 3 long-context prompt markdown files they consume |
| `prompts/` | `glm52_trace_smoke_suite.{json,expanded.jsonl}` (161-prompt multilingual suite, 7 langs × 7 domains) + `glm52_code_switch_suite.expanded.jsonl` (16 bilingual prompts) + `README.md` |
| `reports/` | 37 common-axes `.md` + `.json` artifacts — multilingual routing, code-switch, cross-task/cross-language activation, #4386 investigation (Phases 6 #1-#7), MLA retrieval-pattern study |
| `traces/` | 3 git-tracked common-axes traces (`README.md` + `glm52-coding-en-real-sample.jsonl` + meta sidecar) + gitignored local datasets (`batch/`, `smoke/`, `*.run.log`, MLA retrieval smokes) |
| `phase3_dsa_unblock/` | rejected 3-line DSA indexer patch — `*.txt` artifacts + `README.md` documenting why it can't unlock Story 5's DSA indexer tracing on this model |
| `reap37/` | REAP37 prebuilt MLX-4bit experimental track — 5 scripts + 4 baseline outputs (`scripts/` + `outputs/`) |

## Reproducibility cross-refs

- Output formats produced by these scripts: schema lives in
  `../glm52_kitchen/tracing/schema.py` (`MoeRoutingRecord`,
  `ActivationSummaryRecord`, `RunMetadata`).
- For prune-specific analysis scripts (BI scoring, layer-drop planning),
  see `../layer-level-structured-pruning/scripts/`.
