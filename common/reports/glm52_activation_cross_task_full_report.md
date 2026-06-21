# Cross-task activation analysis

Per-(task, layer, tensor_stem) top-10 channel comparison across 7 tasks and 13 (layer, tensor_stem) groups.

Jaccard = |A ∩ B| / |A ∪ B|. Lower overlap = more task-specific channel selection at that layer.


## Per-layer task-pair Jaccard overlap

| layer | tensor_stem | mean Jaccard | min | max | min pair | max pair |

|---|---|---|---|---|---|---|
| 0 | `l_out` | 0.863 | 0.667 | 1.000 | computer_science↔math | chemistry↔coding |
| 6 | `l_out` | 0.905 | 0.818 | 1.000 | chemistry↔computer_science | chemistry↔coding |
| 12 | `l_out` | 0.948 | 0.818 | 1.000 | chemistry↔physics | chemistry↔coding |
| 18 | `l_out` | 0.861 | 0.818 | 1.000 | chemistry↔coding | chemistry↔physics |
| 24 | `l_out` | 1.000 | 1.000 | 1.000 | chemistry↔coding | chemistry↔coding |
| 30 | `l_out` | 1.000 | 1.000 | 1.000 | chemistry↔coding | chemistry↔coding |
| 36 | `l_out` | 0.835 | 0.818 | 1.000 | chemistry↔coding | chemistry↔physics |
| 42 | `l_out` | 0.767 | 0.538 | 1.000 | computer_science↔cybersecurity | coding↔math |
| 48 | `l_out` | 0.379 | 0.250 | 0.429 | coding↔cybersecurity | chemistry↔coding |
| 54 | `l_out` | 0.397 | 0.250 | 0.667 | cybersecurity↔engineering | computer_science↔math |
| 60 | `l_out` | 0.403 | 0.250 | 0.667 | chemistry↔coding | computer_science↔math |
| 66 | `l_out` | 0.442 | 0.333 | 0.538 | chemistry↔coding | chemistry↔physics |
| 72 | `l_out` | 0.688 | 0.429 | 1.000 | chemistry↔cybersecurity | chemistry↔physics |

## Shared core channels (in ≥half of all tasks)

| layer | tensor_stem | # shared | shared core channels |
|---|---|---|---|
| 0 | `l_out` | 10 | #506, #822, #915, #2232, #2305, #3368, #4386, #4752, #4801, #5434 |
| 6 | `l_out` | 10 | #506, #2232, #2305, #2674, #3203, #3257, #4386, #4801, #5434, #5943 |
| 12 | `l_out` | 10 | #506, #2232, #2305, #2674, #3203, #3257, #4386, #4801, #5655, #5943 |
| 18 | `l_out` | 11 | #506, #2232, #2305, #2674, #3203, #3257, #4386, #4801, #5198, #5655, #5943 |
| 24 | `l_out` | 10 | #506, #2232, #2305, #2588, #3203, #3257, #4386, #4801, #5655, #5943 |
| 30 | `l_out` | 10 | #506, #2232, #2305, #2588, #2674, #3203, #3257, #4386, #4801, #5943 |
| 36 | `l_out` | 9 | #506, #2232, #2305, #2588, #2674, #3203, #4386, #4801, #5943 |
| 42 | `l_out` | 10 | #96, #186, #506, #2232, #2588, #3015, #3203, #4386, #4801, #5809 |
| 48 | `l_out` | 6 | #420, #2232, #3015, #3203, #4386, #4801 |
| 54 | `l_out` | 6 | #1806, #2232, #2800, #3015, #3203, #4386 |
| 60 | `l_out` | 7 | #714, #1806, #2232, #3015, #3203, #4386, #5722 |
| 66 | `l_out` | 8 | #714, #1806, #2842, #2906, #3015, #3203, #4386, #5722 |
| 72 | `l_out` | 10 | #116, #714, #1422, #1806, #2342, #2842, #2906, #3015, #4386, #5722 |

## Task-specific channels (unique to one task)

| layer | tensor_stem | task | # uniq | unique channels |
|---|---|---|---|---|
| 0 | `l_out` | computer_science | 1 | #5652 |
| 6 | `l_out` | computer_science | 1 | #4241 |
| 6 | `l_out` | physics | 1 | #5655 |
| 12 | `l_out` | physics | 1 | #5198 |
| 36 | `l_out` | cybersecurity | 1 | #2800 |
| 36 | `l_out` | engineering | 1 | #4061 |
| 36 | `l_out` | math | 1 | #394 |
| 42 | `l_out` | chemistry | 1 | #2263 |
| 42 | `l_out` | computer_science | 1 | #4749 |
| 42 | `l_out` | cybersecurity | 2 | #279, #2970 |
| 42 | `l_out` | engineering | 1 | #2964 |
| 48 | `l_out` | chemistry | 3 | #1819, #3099, #5207 |
| 48 | `l_out` | coding | 3 | #2906, #4137, #5479 |
| 48 | `l_out` | computer_science | 2 | #1086, #2355 |
| 48 | `l_out` | cybersecurity | 3 | #385, #775, #2971 |
| 48 | `l_out` | engineering | 4 | #1806, #2033, #2589, #5172 |
| 48 | `l_out` | math | 4 | #474, #2480, #4823, #5722 |
| 48 | `l_out` | physics | 2 | #714, #3613 |
| 54 | `l_out` | chemistry | 3 | #2496, #5239, #5627 |
| 54 | `l_out` | coding | 3 | #1467, #2144, #2263 |
| 54 | `l_out` | computer_science | 1 | #2355 |
| 54 | `l_out` | cybersecurity | 5 | #385, #2712, #2971, #5013, #5976 |
| 54 | `l_out` | engineering | 2 | #2508, #2735 |
| 54 | `l_out` | math | 2 | #420, #6086 |
| 54 | `l_out` | physics | 3 | #714, #1738, #5221 |
| 60 | `l_out` | chemistry | 3 | #92, #3738, #5963 |
| 60 | `l_out` | coding | 3 | #1275, #1467, #1509 |
| 60 | `l_out` | cybersecurity | 4 | #2712, #2945, #3546, #5976 |
| 60 | `l_out` | engineering | 3 | #1002, #2735, #4587 |
| 60 | `l_out` | math | 1 | #4494 |
| 60 | `l_out` | physics | 1 | #5221 |
| 66 | `l_out` | chemistry | 3 | #3721, #3738, #5270 |
| 66 | `l_out` | coding | 3 | #1057, #2649, #5636 |
| 66 | `l_out` | computer_science | 2 | #1561, #2436 |
| 66 | `l_out` | cybersecurity | 4 | #501, #2945, #4596, #5976 |
| 66 | `l_out` | engineering | 3 | #1002, #2508, #3835 |
| 66 | `l_out` | math | 3 | #28, #3863, #5809 |
| 66 | `l_out` | physics | 2 | #181, #5221 |
| 72 | `l_out` | coding | 1 | #2649 |
| 72 | `l_out` | computer_science | 1 | #2504 |
| 72 | `l_out` | cybersecurity | 3 | #501, #3641, #5976 |
| 72 | `l_out` | math | 1 | #390 |

## Aggregate (cross-task)

- Overall mean task-pair Jaccard across all layers: **0.730**
- Layers: 0..72

## Per-task unique channel count (sum across layers)

| task | # unique channels |
|---|---|
| cybersecurity | 22 |
| engineering | 14 |
| chemistry | 13 |
| coding | 13 |
| math | 12 |
| physics | 10 |
| computer_science | 9 |

---

# Cross-language activation analysis

Pairwise top-10 Jaccard across 7 languages (redone symmetry to the task analysis above).


## Per-layer language-pair Jaccard overlap

| layer | tensor_stem | mean Jaccard | min | max | min pair | max pair |

|---|---|---|---|---|---|---|
| 0 | `l_out` | 0.861 | 0.667 | 1.000 | de↔zh | de↔es |
| 6 | `l_out` | 0.948 | 0.818 | 1.000 | de↔en | de↔es |
| 12 | `l_out` | 0.879 | 0.818 | 1.000 | de↔en | en↔zh |
| 18 | `l_out` | 0.879 | 0.818 | 1.000 | de↔en | en↔es |
| 24 | `l_out` | 0.948 | 0.818 | 1.000 | de↔en | de↔es |
| 30 | `l_out` | 1.000 | 1.000 | 1.000 | de↔en | de↔en |
| 36 | `l_out` | 0.898 | 0.667 | 1.000 | fr↔zh | de↔en |
| 42 | `l_out` | 0.808 | 0.667 | 1.000 | en↔fr | de↔es |
| 48 | `l_out` | 0.521 | 0.333 | 0.667 | de↔zh | en↔es |
| 54 | `l_out` | 0.507 | 0.333 | 0.667 | fr↔zh | en↔es |
| 60 | `l_out` | 0.478 | 0.333 | 0.538 | de↔zh | de↔es |
| 66 | `l_out` | 0.507 | 0.333 | 0.667 | fr↔zh | de↔en |
| 72 | `l_out` | 0.527 | 0.333 | 0.667 | es↔fr | de↔it |

_Approximation: top-K channel overlap is a lower-bound interpretability signal, not a full activation trace. See `GLM52_SESSION_MEMORY.md` for scope._
