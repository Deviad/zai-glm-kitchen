# Phase 6 — channel #2232 focus investigation

Scanned 94,991 activation records across 161 trace files. Channel #2232 appeared in 35,960 of them (37.9%) and was the rank-1 (highest-magnitude) channel in 1,028 (1.1%).


## Rank distribution (when #channel is in top-K)

| rank | count | % |
|---|---|---|
| 1 | 1,028 | 2.9% |
| 2 | 6,252 | 17.4% |
| 3 | 5,706 | 15.9% |
| 4 | 4,068 | 11.3% |
| 5 | 4,103 | 11.4% |
| 6 | 3,399 | 9.5% |
| 7 | 2,958 | 8.2% |
| 8 | 2,309 | 6.4% |
| 9 | 1,724 | 4.8% |
| 10 | 1,265 | 3.5% |
| 11 | 868 | 2.4% |
| 12 | 680 | 1.9% |
| 13 | 609 | 1.7% |
| 14 | 550 | 1.5% |
| 15 | 441 | 1.2% |

## Per-layer magnitude

| layer | n | mean | std | min | max |
|---|---|---|---|---|---|
| 0 | 3,579 | -0.046619 | 0.019247 | -0.126631 | 0.045911 |
| 6 | 4,300 | -0.016922 | 0.130648 | -0.207518 | 0.85246 |
| 12 | 4,363 | -0.02138 | 0.137939 | -0.599883 | 0.381208 |
| 18 | 4,186 | -0.107964 | 0.283754 | -0.874103 | 2.52606 |
| 24 | 4,025 | 0.136063 | 0.919626 | -1.42599 | 5.47828 |
| 30 | 4,493 | 3.074868 | 14.387765 | -2.76682 | 74.7339 |
| 36 | 4,152 | 3.969516 | 16.818595 | -4.43679 | 81.8627 |
| 42 | 3,751 | 4.433899 | 18.055798 | -4.83366 | 83.6477 |
| 48 | 2,180 | 5.890823 | 20.328648 | -4.17577 | 74.8384 |
| 54 | 380 | 28.961192 | 26.367805 | -2.99368 | 59.4246 |
| 60 | 316 | 32.541402 | 25.643057 | -3.07077 | 57.9106 |
| 66 | 189 | 48.036293 | 20.179497 | -4.87361 | 56.9979 |
| 72 | 46 | -4.704951 | 2.149294 | -6.99321 | 3.09169 |

## Mean magnitude by (task, phase)

| task | phase | n | mean | std | min | max |
|---|---|---|---|---|---|---|
| chemistry | generation | 815 | 0.248487 | 1.105297 | -6.58784 | 4.35688 |
| chemistry | prefill | 3,301 | 3.30461 | 15.107801 | -6.81314 | 83.6051 |
| coding | generation | 1,121 | 0.223288 | 1.893244 | -4.83076 | 31.5323 |
| coding | prefill | 5,300 | 2.91073 | 13.7615 | -6.99321 | 83.6174 |
| computer_science | generation | 843 | 0.094519 | 1.586373 | -5.94721 | 20.2985 |
| computer_science | prefill | 3,011 | 4.04987 | 15.856374 | -6.4512 | 83.603 |
| cybersecurity | generation | 812 | 0.226914 | 0.833544 | -5.08009 | 5.38775 |
| cybersecurity | prefill | 4,739 | 2.205203 | 12.641487 | -6.58173 | 83.6205 |
| engineering | generation | 906 | 0.384306 | 1.056285 | -2.94235 | 4.27539 |
| engineering | prefill | 4,213 | 2.691198 | 13.468786 | -5.75817 | 83.6205 |
| math | generation | 901 | 0.650916 | 1.944767 | -3.64675 | 28.9977 |
| math | prefill | 4,012 | 2.796405 | 13.852432 | -4.83366 | 83.6477 |
| physics | generation | 1,183 | 0.43896 | 1.82295 | -3.79457 | 30.8324 |
| physics | prefill | 4,803 | 3.326266 | 14.702836 | -4.95996 | 83.6205 |

## Mean magnitude by (language, phase)

| language | phase | n | mean | std | min | max |
|---|---|---|---|---|---|---|
| de | generation | 936 | 0.157684 | 1.01353 | -5.08009 | 4.08515 |
| de | prefill | 4,985 | 2.428349 | 13.088914 | -6.15945 | 83.5878 |
| en | generation | 1,003 | 0.616526 | 1.188141 | -6.58784 | 5.36399 |
| en | prefill | 3,102 | 4.391762 | 16.449645 | -6.81314 | 83.6477 |
| es | generation | 1,001 | 0.32582 | 1.068668 | -2.67552 | 5.38775 |
| es | prefill | 4,376 | 2.816172 | 13.845521 | -6.58173 | 83.5586 |
| fr | generation | 943 | 0.227965 | 1.020565 | -5.94721 | 4.00269 |
| fr | prefill | 4,876 | 2.370301 | 13.08275 | -6.99321 | 83.6051 |
| it | generation | 905 | 0.25087 | 0.97996 | -5.08414 | 3.81602 |
| it | prefill | 4,713 | 2.610025 | 13.389441 | -4.68896 | 83.5523 |
| pt | generation | 923 | 0.377581 | 2.591951 | -3.24768 | 31.5323 |
| pt | prefill | 4,330 | 2.798655 | 13.797006 | -4.59581 | 83.6051 |
| zh | generation | 870 | 0.331118 | 2.282312 | -3.79457 | 30.8324 |
| zh | prefill | 2,997 | 4.499267 | 16.320719 | -4.13809 | 83.3579 |

## Position distribution when #channel is rank-1
(token position as fraction of prompt_len — prefill phase)

| task | 0-10% | 10-25% | 25-50% | 50-75% | 75-90% | 90-100% | other |
|---|---|---|---|---|---|---|---|
| chemistry | 14 | 19 | 18 | 14 | 7 | 1 | 0 |
| coding | 17 | 24 | 25 | 39 | 21 | 6 | 0 |
| computer_science | 24 | 36 | 19 | 17 | 3 | 1 | 0 |
| cybersecurity | 19 | 41 | 45 | 22 | 8 | 1 | 0 |
| engineering | 16 | 32 | 46 | 16 | 4 | 1 | 0 |
| math | 34 | 29 | 19 | 12 | 5 | 1 | 0 |
| physics | 27 | 44 | 29 | 14 | 8 | 2 | 0 |

## Co-firing channels (top-20, when #channel is rank-1)
These channels appear alongside #2232 in the same token's top-K.

| channel | co-occurrences |
|---|---|
| #4386 | 576 |
| #506 | 515 |
| #2588 | 471 |
| #4801 | 469 |
| #3203 | 456 |
| #5943 | 285 |
| #2800 | 275 |
| #4923 | 269 |
| #96 | 269 |
| #186 | 268 |
| #2293 | 265 |
| #1431 | 265 |
| #1362 | 265 |
| #4196 | 264 |
| #274 | 262 |
| #2305 | 181 |
| #3424 | 171 |
| #2674 | 144 |
| #4454 | 128 |
| #848 | 118 |

_Approximation: channel magnitudes are only collected when the target channel is already in that token's top-K (default 15). Records where the channel ranks lower than top-K are not visible to this analyzer. See GLM52_SESSION_MEMORY.md for scope._
