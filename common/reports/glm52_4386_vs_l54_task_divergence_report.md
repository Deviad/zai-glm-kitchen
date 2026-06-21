# Phase 6 — #4386's role in Phase 5b's L54 task divergence

Evidence backing `GLM52_SESSION_MEMORY.md` Phase 6 follow-up #2 (2026-06-20).

## Method

Two cross-checks on existing 161-prompt trace data:

1. **Token-0 #4386 magnitude per task at L54** — if it differs greatly per task,
   #4386 is the divergence channel. If it is uniform, #4386 is a constant backdrop.

2. **Phase 5b's task_specific sets at L54** — does #4386 appear in any task's
   'unique channels' list? If yes, divergence-driving. If no, backdrop.

## Cross-check 1 — Token-0 #4386 magnitude per task at L54

All rank-1 #4386 records at L54 prefill, grouped by task and position
(tok0 = first prompt token, tok1+ = all others).

| task | position | n | min | mean | max | median |
|---|---|---|---|---|---|---|
| chemistry | tok0 | 21 | -225.372 | -224.570 | -223.444 | -224.836 |
| chemistry | tok1+ | 648 | -27.340 | 19.813 | 72.406 | 20.023 |
| coding | tok0 | 28 | -225.377 | -224.171 | -222.449 | -223.788 |
| coding | tok1+ | 1008 | -28.727 | 18.994 | 66.660 | 19.560 |
| computer_science | tok0 | 21 | -225.355 | -224.453 | -220.479 | -224.874 |
| computer_science | tok1+ | 605 | -32.918 | 19.086 | 70.161 | 19.774 |
| cybersecurity | tok0 | 21 | -225.438 | -224.492 | -221.407 | -225.219 |
| cybersecurity | tok1+ | 924 | -27.210 | 23.932 | 55.744 | 24.641 |
| engineering | tok0 | 21 | -225.377 | -224.566 | -222.539 | -224.881 |
| engineering | tok1+ | 855 | -29.821 | 21.809 | 51.001 | 22.463 |
| math | tok0 | 21 | -225.502 | -224.111 | -216.963 | -224.547 |
| math | tok1+ | 726 | -28.590 | 19.501 | 63.694 | 20.193 |
| physics | tok0 | 28 | -225.377 | -224.715 | -223.233 | -225.152 |
| physics | tok1+ | 923 | -32.921 | 21.656 | 58.408 | 22.684 |

**Cross-task token-0 saturation**: medians -223.8 to -225.2, <1% variation.
All tasks saturate #4386 to essentially the same value on token 0.

## Cross-check 2 — #4386 presence in Phase 5b's task_specific sets

Direct check against the Phase 5b per-layer task_specific channel lists,
covering every deep layer L24-L72.

| layer | mean_jaccard | #4386 in shared_core | #4386 in any task_specific |
|---|---|---|---|
| 24 | 1.000 | True | False |
| 30 | 1.000 | True | False |
| 36 | 0.835 | True | False |
| 42 | 0.767 | True | False |
| 48 | 0.379 | True | False |
| 54 | 0.397 | True | False |
| 60 | 0.403 | True | False |
| 66 | 0.442 | True | False |
| 72 | 0.688 | True | False |

## L54 detail — task_specific breakdown

| task | # task-specific channels | #4386 in set? |
|---|---|---|
| chemistry | 3 | False |
| coding | 3 | False |
| computer_science | 1 | False |
| cybersecurity | 5 | False |
| engineering | 2 | False |
| math | 2 | False |
| physics | 3 | False |

**Shared core at L54** (6 channels, in ≥4/7 tasks): [1806, 2232, 2800, 3015, 3203, 4386]

## Conclusion

Phase 5b's L54 task divergence comes **DESPITE #4386, not because of it**.

- #4386 is in shared core at every deep layer L24-L72 (universally present)
- #4386 is NEVER in any task_specific set at any deep layer
- Token-0 saturation magnitude varies <1% across tasks at L54

#4386 is a constant backdrop. Phase 5b's L54 divergence is driven entirely
by OTHER channels — cybersecurity's 5 task-specific channels at L54,
chemistry's 3, etc. — all diverging against #4386's constant presence.

## Provenance

- Phase 5b summary JSON: reports/glm52_activation_cross_task_full_summary.json
- Trace dataset: traces/batch/activation_full_161/*.jsonl
- Method: single Python script, no new C++ runs
