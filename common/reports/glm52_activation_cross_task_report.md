# Cross-task activation analysis

Per-(task, layer, tensor_stem) top-10 channel comparison across 7 tasks and 13 (layer, tensor_stem) groups.

Jaccard = |A ∩ B| / |A ∪ B|. Lower overlap = more task-specific channel selection at that layer.


## Per-layer task-pair Jaccard overlap

| layer | tensor_stem | mean Jaccard | min | max | min pair | max pair |

|---|---|---|---|---|---|---|
| 0 | `l_out` | 0.784 | 0.667 | 1.000 | coding↔computer_science | chemistry↔engineering |
| 6 | `l_out` | 0.817 | 0.667 | 1.000 | chemistry↔coding | coding↔computer_science |
| 12 | `l_out` | 0.870 | 0.818 | 1.000 | chemistry↔coding | chemistry↔cybersecurity |
| 18 | `l_out` | 0.861 | 0.818 | 1.000 | chemistry↔coding | chemistry↔physics |
| 24 | `l_out` | 0.948 | 0.818 | 1.000 | chemistry↔coding | coding↔computer_science |
| 30 | `l_out` | 1.000 | 1.000 | 1.000 | chemistry↔coding | chemistry↔coding |
| 36 | `l_out` | 0.677 | 0.538 | 0.818 | chemistry↔computer_science | coding↔cybersecurity |
| 42 | `l_out` | 0.392 | 0.250 | 0.538 | chemistry↔cybersecurity | coding↔computer_science |
| 48 | `l_out` | 0.278 | 0.250 | 0.333 | chemistry↔coding | chemistry↔physics |
| 54 | `l_out` | 0.208 | 0.176 | 0.333 | chemistry↔coding | coding↔math |
| 60 | `l_out` | 0.258 | 0.176 | 0.429 | chemistry↔coding | computer_science↔engineering |
| 66 | `l_out` | 0.282 | 0.250 | 0.429 | chemistry↔coding | chemistry↔physics |
| 72 | `l_out` | 0.514 | 0.429 | 0.667 | chemistry↔cybersecurity | chemistry↔computer_science |

## Shared core channels (in ≥half of all tasks)

| layer | tensor_stem | # shared | shared core channels |
|---|---|---|---|
| 0 | `l_out` | 10 | #506, #822, #915, #2232, #2305, #3368, #4386, #4752, #4801, #5434 |
| 6 | `l_out` | 9 | #506, #2232, #2305, #3203, #3257, #4386, #4801, #5434, #5943 |
| 12 | `l_out` | 10 | #506, #2232, #2305, #2674, #3203, #3257, #4386, #4801, #5655, #5943 |
| 18 | `l_out` | 11 | #506, #2232, #2305, #2674, #3203, #3257, #4386, #4801, #5198, #5655, #5943 |
| 24 | `l_out` | 10 | #506, #2232, #2305, #2588, #3203, #3257, #4386, #4801, #5655, #5943 |
| 30 | `l_out` | 10 | #506, #2232, #2305, #2588, #2674, #3203, #3257, #4386, #4801, #5943 |
| 36 | `l_out` | 9 | #506, #2232, #2305, #2588, #2674, #3203, #4386, #4801, #5943 |
| 42 | `l_out` | 7 | #506, #2232, #3015, #3203, #4386, #4801, #5809 |
| 48 | `l_out` | 4 | #2232, #3015, #3203, #4386 |
| 54 | `l_out` | 3 | #3015, #3203, #4386 |
| 60 | `l_out` | 4 | #1806, #3015, #3203, #4386 |
| 66 | `l_out` | 5 | #1806, #2906, #3015, #3203, #4386 |
| 72 | `l_out` | 7 | #116, #1422, #1806, #2842, #2906, #3015, #4386 |

## Task-specific channels (unique to one task)

| layer | tensor_stem | task | # uniq | unique channels |
|---|---|---|---|---|
| 0 | `l_out` | computer_science | 1 | #5652 |
| 0 | `l_out` | math | 1 | #5943 |
| 0 | `l_out` | physics | 1 | #3257 |
| 6 | `l_out` | chemistry | 1 | #5198 |
| 12 | `l_out` | coding | 1 | #2588 |
| 12 | `l_out` | computer_science | 1 | #5198 |
| 12 | `l_out` | physics | 1 | #2012 |
| 24 | `l_out` | chemistry | 1 | #2674 |
| 36 | `l_out` | chemistry | 2 | #1257, #2856 |
| 36 | `l_out` | coding | 1 | #2800 |
| 36 | `l_out` | computer_science | 2 | #1950, #4749 |
| 36 | `l_out` | cybersecurity | 1 | #1667 |
| 36 | `l_out` | engineering | 2 | #3239, #4061 |
| 36 | `l_out` | math | 1 | #622 |
| 42 | `l_out` | chemistry | 3 | #2188, #3011, #3099 |
| 42 | `l_out` | coding | 2 | #2800, #6113 |
| 42 | `l_out` | computer_science | 3 | #714, #1950, #4749 |
| 42 | `l_out` | cybersecurity | 4 | #15, #538, #4884, #6038 |
| 42 | `l_out` | engineering | 3 | #186, #2361, #4333 |
| 42 | `l_out` | math | 4 | #1561, #2228, #2480, #3199 |
| 42 | `l_out` | physics | 4 | #1810, #2588, #2655, #3330 |
| 48 | `l_out` | chemistry | 5 | #2188, #2296, #2301, #3099, #5207 |
| 48 | `l_out` | coding | 4 | #186, #420, #2322, #5086 |
| 48 | `l_out` | computer_science | 3 | #714, #1086, #3710 |
| 48 | `l_out` | cybersecurity | 4 | #501, #538, #2971, #3957 |
| 48 | `l_out` | engineering | 4 | #350, #3907, #5172, #5817 |
| 48 | `l_out` | math | 5 | #474, #1561, #2480, #3583, #4378 |
| 48 | `l_out` | physics | 3 | #158, #482, #4709 |
| 54 | `l_out` | chemistry | 7 | #264, #2397, #3738, #4830, #4864, #5152, #5627 |
| 54 | `l_out` | coding | 4 | #181, #668, #2943, #5406 |
| 54 | `l_out` | computer_science | 5 | #2355, #2671, #3183, #3664, #5151 |
| 54 | `l_out` | cybersecurity | 6 | #501, #803, #2712, #2971, #4596, #5330 |
| 54 | `l_out` | engineering | 6 | #2108, #2361, #2508, #2735, #3900, #4613 |
| 54 | `l_out` | math | 4 | #1420, #2480, #4378, #5319 |
| 54 | `l_out` | physics | 5 | #550, #4329, #4424, #5207, #5864 |
| 60 | `l_out` | chemistry | 7 | #264, #2296, #2397, #3738, #5059, #5406, #5860 |
| 60 | `l_out` | coding | 3 | #2212, #2943, #4479 |
| 60 | `l_out` | computer_science | 2 | #2868, #5722 |
| 60 | `l_out` | cybersecurity | 5 | #501, #2712, #2971, #4596, #5885 |
| 60 | `l_out` | engineering | 4 | #2735, #3546, #5042, #6101 |
| 60 | `l_out` | math | 3 | #983, #2480, #4494 |
| 60 | `l_out` | physics | 5 | #425, #550, #2361, #4424, #4709 |
| 66 | `l_out` | chemistry | 4 | #2397, #3530, #3738, #5860 |
| 66 | `l_out` | coding | 5 | #2436, #3126, #3941, #4479, #5346 |
| 66 | `l_out` | computer_science | 4 | #714, #1561, #1928, #5172 |
| 66 | `l_out` | cybersecurity | 6 | #501, #2712, #2945, #4596, #5235, #5885 |
| 66 | `l_out` | engineering | 5 | #2108, #2735, #3045, #3838, #5042 |
| 66 | `l_out` | math | 6 | #280, #893, #3721, #3863, #4378, #5809 |
| 66 | `l_out` | physics | 4 | #526, #5221, #5722, #5933 |
| 72 | `l_out` | chemistry | 2 | #2397, #5579 |
| 72 | `l_out` | coding | 3 | #349, #2342, #4479 |
| 72 | `l_out` | cybersecurity | 3 | #501, #2712, #3641 |
| 72 | `l_out` | engineering | 2 | #3080, #4472 |
| 72 | `l_out` | math | 2 | #390, #4494 |
| 72 | `l_out` | physics | 2 | #774, #1006 |

## Aggregate (cross-task)

- Overall mean task-pair Jaccard across all layers: **0.607**
- Layers: 0..72

## Per-task unique channel count (sum across layers)

| task | # unique channels |
|---|---|
| chemistry | 32 |
| cybersecurity | 29 |
| math | 26 |
| engineering | 26 |
| physics | 25 |
| coding | 23 |
| computer_science | 21 |

---

# Cross-language activation analysis

Pairwise top-10 Jaccard across 7 languages (redone symmetry to the task analysis above).


## Per-layer language-pair Jaccard overlap

| layer | tensor_stem | mean Jaccard | min | max | min pair | max pair |

|---|---|---|---|---|---|---|
| 0 | `l_out` | 0.861 | 0.667 | 1.000 | de↔zh | de↔es |
| 6 | `l_out` | 0.870 | 0.818 | 1.000 | de↔en | de↔es |
| 12 | `l_out` | 0.913 | 0.818 | 1.000 | de↔es | de↔en |
| 18 | `l_out` | 0.879 | 0.818 | 1.000 | de↔en | de↔es |
| 24 | `l_out` | 0.948 | 0.818 | 1.000 | de↔en | de↔es |
| 30 | `l_out` | 1.000 | 1.000 | 1.000 | de↔en | de↔en |
| 36 | `l_out` | 0.879 | 0.818 | 1.000 | de↔en | de↔fr |
| 42 | `l_out` | 0.667 | 0.429 | 0.818 | de↔en | de↔it |
| 48 | `l_out` | 0.373 | 0.250 | 0.538 | de↔zh | es↔fr |
| 54 | `l_out` | 0.485 | 0.333 | 0.667 | en↔fr | de↔fr |
| 60 | `l_out` | 0.461 | 0.333 | 0.667 | de↔pt | en↔es |
| 66 | `l_out` | 0.389 | 0.250 | 0.667 | en↔zh | es↔it |
| 72 | `l_out` | 0.495 | 0.333 | 0.667 | fr↔zh | de↔en |

_Approximation: top-K channel overlap is a lower-bound interpretability signal, not a full activation trace. See `GLM52_SESSION_MEMORY.md` for scope._
