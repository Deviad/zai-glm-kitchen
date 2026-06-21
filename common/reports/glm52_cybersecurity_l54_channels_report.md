# Phase 6 follow-up #5 — Cybersecurity's L54 task-specific channels

Evidence backing `GLM52_SESSION_MEMORY.md` Phase 6 follow-up #5 (2026-06-20).

## Question

Phase 6 #2 established that Phase 5b's L54 task divergence comes from
task-specific channels OTHER than #4386. Cybersecurity has the most unique
channels at L54 (5: #385, #2712, #2971, #5013, #5976).

Are these genuine task-semantics, or just more positional/structural markers
miscategorized as task-specific?

## Method

Run `analyze_channel_focus.py` on each of cybersecurity's 5 L54 unique
channels. Compare their behavior to known markers (#4386, #2232, #506, #2305).

A genuine task-semantics channel should:
1. Lack the token-0 saturation pattern (low tok0/10 ratio)
2. Have small overall magnitudes (not ~226 like #4386)
3. Fire in actually-cybersecurity contexts, not just structural positions

## Source channel list (from Phase 5b L54 task_specific sets)

```
ALL tasks' L54 unique channels:
  chemistry         : [2496, 5239, 5627]                                — 3 channels
  coding            : [1467, 2144, 2263]                                — 3 channels
  computer_science  : [2355]                                            — 1 channel
  cybersecurity     : [385, 2712, 2971, 5013, 5976]                    — 5 channels (most)
  engineering       : [2508, 2735]                                      — 2 channels
  math              : [420, 6086]                                       — 2 channels
  physics           : [714, 1738, 5221]                                 — 3 channels
```

## Per-channel analysis — top-5 highest-|magnitude| records

For each channel, the analyzer's top-5 highest |magnitude| events across all
161 trace files (any phase, any layer, any token) are listed below.

### Channel #385

| rank | |mag| | magnitude | layer | tok_idx | phase | task | lang |
|---|---|---|---|---|---|---|---|
| 1 | 6.49 | +6.49 | L72 | 25 | prefill | cybersecurity | en |
| 2 | 6.24 | +6.24 | L72 | 11 | prefill | physics | zh |
| 3 | 5.96 | +5.96 | L72 | 22 | prefill | coding | it |
| 4 | 5.71 | +5.71 | L72 | 50 | prefill | cybersecurity | es |
| 5 | 5.68 | +5.68 | L72 | 15 | prefill | coding | en |

- n records with ch: 438 (0.5%)
- rank-1: 1 (0.0%)
- tok0 count in top-10: **0/10 (not a marker)**
- Profile: cross-task — fires for cybersecurity + physics + coding at L72. Less narrowly cybersecurity-specific.

### Channel #2712

| rank | |mag| | magnitude | layer | tok_idx | phase | task | lang |
|---|---|---|---|---|---|---|---|
| 1 | 8.10 | +8.10 | L72 | 45 | prefill | cybersecurity | it |
| 2 | 7.94 | +7.94 | L72 | 34 | prefill | cybersecurity | zh |
| 3 | 7.90 | +7.90 | L72 | 38 | prefill | cybersecurity | zh |
| 4 | 7.69 | +7.69 | L72 | 19 | prefill | cybersecurity | zh |
| 5 | 7.53 | +7.53 | L72 | 24 | prefill | cybersecurity | en |

- n records with ch: 450 (0.5%)
- rank-1: 4 (0.0%)
- tok0 count in top-10: **0/10 (not a marker)**
- Profile: STRONGLY cybersecurity-specific (4/5 top records are cybersecurity). Genuine task-semantics.

### Channel #2971

| rank | |mag| | magnitude | layer | tok_idx | phase | task | lang |
|---|---|---|---|---|---|---|---|
| 1 | 6.05 | +6.05 | L72 | 41 | prefill | cybersecurity | zh |
| 2 | 5.86 | +5.86 | L72 | 41 | prefill | cybersecurity | de |
| 3 | 5.44 | +5.44 | L66 | 41 | prefill | cybersecurity | zh |
| 4 | 5.38 | +5.38 | L72 | 29 | prefill | cybersecurity | pt |
| 5 | 5.28 | -5.28 | L72 | 7 | prefill | physics | fr |

- n records with ch: 254 (0.3%)
- rank-1: 3 (0.0%)
- tok0 count in top-10: **0/10 (not a marker)**
- Profile: STRONGLY cybersecurity-specific (4/5 top records are cybersecurity). Genuine task-semantics.

### Channel #5013

| rank | |mag| | magnitude | layer | tok_idx | phase | task | lang |
|---|---|---|---|---|---|---|---|
| 1 | 6.83 | -6.83 | L72 | 56 | generation | cybersecurity | de |
| 2 | 6.20 | -6.20 | L72 | 4 | prefill | chemistry | de |
| 3 | 5.64 | +5.64 | L72 | 43 | prefill | engineering | it |
| 4 | 5.40 | -5.40 | L72 | 25 | prefill | coding | de |
| 5 | 5.40 | -5.40 | L72 | 28 | prefill | cybersecurity | en |

- n records with ch: 320 (0.3%)
- rank-1: 1 (0.0%)
- tok0 count in top-10: **0/10 (not a marker)**
- Profile: MOSTLY NEGATIVE polarity. Cross-task top records (chemistry, engineering, coding). Possible general "risk/hazard" or "inhibition" channel.

### Channel #5976

| rank | |mag| | magnitude | layer | tok_idx | phase | task | lang |
|---|---|---|---|---|---|---|---|
| 1 | 8.74 | +8.74 | L72 | 20 | prefill | cybersecurity | fr |
| 2 | 8.60 | +8.60 | L72 | 8 | prefill | cybersecurity | pt |
| 3 | 8.46 | +8.46 | L72 | 24 | prefill | cybersecurity | fr |
| 4 | 8.26 | +8.26 | L72 | 8 | prefill | cybersecurity | fr |
| 5 | 8.00 | +8.00 | L72 | 29 | prefill | cybersecurity | fr |

- n records with ch: 760 (0.8%)
- rank-1: 7 (0.0%)
- tok0 count in top-10: **0/10 (not a marker)**
- Profile: TIGHTLY cybersecurity-only (5/5 top records). All Romance-language prompts (fr, pt). Most narrowly-specific channel.

## Summary table

| Channel | n_recs_with | rank-1 | peak_layer | peak_mean | tok0/10 | Top-5 task breakdown |
|---|---|---|---|---|---|---|
| 385  | 438 (0.5%) | 0.0% | L72 | +4.79 | 0/10 | cybersecurity + physics + coding |
| 2712 | 450 (0.5%) | 0.0% | L72 | +5.82 | 0/10 | **4/5 cybersecurity** |
| 2971 | 254 (0.3%) | 0.0% | L66/L72 | +3.83 | 0/10 | **4/5 cybersecurity** |
| 5013 | 320 (0.3%) | 0.0% | L72 | -4.35 | 0/10 | cross-task, mostly negative |
| 5976 | 760 (0.8%) | 0.0% | L72 | +5.66 | 0/10 | **5/5 cybersecurity (all Romance-language)** |

Compare vs markers (Phase 6 #1, #3):
- #4386: 95% top-K appearance, 85.5% rank-1, peak magnitude 229, 10/10 tok0
- Markers consistently: peak >35, tok0/10 = 10/10

## Conclusion

1. **All 5 channels lack the marker pattern.** All rank-1 rates are ~0% (vs #4386's
   85.5%). Tok0/10 is 0/10 across all 5 (vs markers' 10/10). Peak magnitudes are
   4-9 (vs #4386's 229, vs even #2232's 84). **These are NOT marker channels.**

2. **3 of 5 are clearly cybersecurity-specific content channels.**
   - #2712: 4/5 top records cybersecurity (zh, it, en, es)
   - #2971: 4/5 top records cybersecurity (zh, de, pt)
   - #5976: 5/5 top records cybersecurity (fr, pt) — the most narrowly cybersecurity-specific

3. **2 of 5 have broader uses.** #385 fires for cybersecurity + physics + coding
   in roughly equal proportions — possible general "adversarial" channel. #5013
   is mostly negative-polarity with cross-task top records — possible
   "inhibition" channel.

4. **Peak layer is L72 for 4 of 5.** Even though they're classified as "L54
   task_specific" (frequent at L54 in cybersecurity top-Ks), their highest-magnitude
   events are at L72 (the output layer). Cybersecurity content signal accumulates
   through the model to peak at the output.

This confirms Phase 6 #2: Phase 5b's L54 task divergence is driven by genuine
task-specific content channels — a constellation of small-magnitude channels
(#385, #2712, #2971, #5013, #5976) firing for cybersecurity content, against
#4386's constant -225 backdrop. The model's task-specific processing is genuinely
localized to small-magnitude channels at depth.

## Provenance

- Source: Phase 5b's `reports/glm52_activation_cross_task_full_summary.json` (L54 task_specific sets)
- Trace dataset: `traces/batch/activation_full_161/*.jsonl` (161 prompts)
- Method: 5 invocations of `scripts/tracing/analyze_channel_focus.py --channel N`, then post-hoc top-K extraction.
- No new C++ runs.
