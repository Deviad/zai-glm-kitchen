# GLM-5.2 MoE Expert-Routing Trace Report

## Reproducibility provenance

All runs in this report traced the same model. Re-run with:

```text
/Users/spotted/projects/llama.cpp/build-metal/bin/llama-trace-moe --model /Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/GLM-5.2-mixed-00001-of-00009.gguf -ngl 999 --ctx-size 32768 --predict 24 --temp 0.0 --batch-size 32768 --file long_coding_task_20k_retrieval_prompt.md
```

- Model: `GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest`
- prompt_sha256 (this run): `18b6c37f408cdce7e3c664bae9768031faabe02e67d900b5b308fb33daa7b978`
- model_sha256_prefix (first 1 MiB): `78a23335f717461a`
- model_total_size_bytes: **249186991232** (232.1 GiB)
- Run window: `2026-06-20T21:32:02Z` → `2026-06-20T21:37:41Z` (UTC)
- Speed: **0.90 gen tok/s**, 61.62 prefill tok/s (24 gen tokens / 18745 prompt tokens)
- n_expert_total: **256** (total routed experts per MoE layer)
- Sample run_id: `retr_longctx-en-1781991130`

- Records traced: **1388931** across **1** run(s)
- Layers seen: **76** (0..77)
- Experts per routing event: up to **8** of **256** total
- Tasks: `coding`
- Languages: `en`

## Top experts by task and layer
### Task: `coding`
| layer | top experts (id×count) |
|---|---|
| 3 | #132×591, #78×590, #75×590, #47×589 |
| 4 | #206×591, #184×590, #124×589, #32×589 |
| 5 | #121×591, #142×589, #179×589, #149×589 |
| 6 | #105×592, #221×592, #164×591, #219×590 |
| 7 | #243×590, #169×590, #111×589, #135×589 |
| 8 | #188×591, #27×591, #64×591, #252×590 |
| 9 | #52×596, #160×595, #33×592, #177×590 |
| 10 | #118×592, #178×591, #192×591, #71×591 |
| 11 | #180×593, #0×591, #184×591, #122×591 |
| 12 | #145×596, #99×595, #214×595, #236×592 |
| 13 | #13×594, #139×592, #223×590, #48×590 |
| 14 | #147×593, #103×591, #135×590, #100×590 |
| 15 | #57×595, #158×592, #59×592, #63×591 |
| 16 | #152×600, #12×596, #70×593, #67×593 |
| 17 | #8×599, #174×595, #184×594, #161×592 |
| 18 | #53×606, #76×601, #122×598, #226×593 |
| 19 | #239×600, #16×600, #69×596, #62×594 |
| 20 | #195×605, #114×599, #56×598, #120×596 |
| 21 | #78×598, #197×595, #168×595, #176×592 |
| 22 | #193×605, #224×596, #239×593, #81×592 |
| 23 | #153×595, #2×593, #20×592, #1×592 |
| 24 | #239×597, #198×593, #131×591, #102×591 |
| 25 | #187×592, #87×591, #131×591, #91×591 |
| 26 | #10×596, #3×594, #104×592, #0×592 |
| 27 | #103×600, #181×594, #243×594, #188×593 |
| 28 | #33×596, #84×595, #11×593, #23×592 |
| 29 | #28×597, #205×597, #39×595, #152×595 |
| 30 | #63×596, #169×596, #255×594, #0×592 |
| 31 | #8×595, #90×594, #207×594, #230×594 |
| 32 | #23×609, #26×595, #55×593, #92×592 |
| 33 | #201×598, #37×598, #20×594, #221×594 |
| 34 | #177×600, #222×597, #65×595, #44×594 |
| 35 | #100×605, #188×601, #173×596, #176×593 |
| 36 | #104×605, #105×604, #35×598, #42×596 |
| 37 | #96×601, #52×598, #144×597, #182×595 |
| 38 | #148×601, #98×601, #38×596, #73×596 |
| 39 | #190×605, #104×604, #138×598, #194×594 |
| 40 | #146×609, #11×604, #46×599, #70×595 |
| 41 | #146×609, #177×606, #155×596, #210×594 |
| 42 | #136×605, #112×599, #5×596, #68×592 |
| 43 | #228×606, #168×597, #187×597, #91×595 |
| 44 | #196×597, #71×596, #177×593, #143×593 |
| 45 | #2×596, #216×596, #161×595, #29×594 |
| 46 | #102×599, #82×596, #35×596, #183×595 |
| 47 | #87×603, #121×600, #236×598, #207×594 |
| 48 | #130×600, #92×595, #38×595, #114×595 |
| 49 | #41×595, #72×595, #133×593, #112×593 |
| 50 | #233×599, #171×598, #164×598, #113×596 |
| 51 | #193×595, #145×594, #80×593, #190×592 |
| 52 | #144×594, #61×593, #2×592, #213×592 |
| 53 | #43×595, #154×595, #51×593, #212×592 |
| 54 | #3×594, #231×594, #17×593, #177×592 |
| 55 | #87×596, #203×593, #152×593, #167×592 |
| 56 | #155×598, #127×594, #175×592, #241×591 |
| 57 | #250×595, #141×594, #235×593, #41×592 |
| 58 | #27×595, #174×593, #177×593, #128×591 |
| 59 | #87×593, #5×592, #207×592, #168×591 |
| 60 | #53×596, #164×595, #188×594, #32×593 |
| 61 | #241×596, #163×595, #142×592, #168×592 |
| 62 | #234×597, #217×594, #61×593, #119×592 |
| 63 | #143×593, #254×592, #104×592, #68×591 |
| 64 | #122×595, #63×593, #82×593, #163×592 |
| 65 | #62×595, #0×592, #199×592, #39×591 |
| 66 | #248×595, #46×592, #206×592, #146×591 |
| 67 | #253×595, #22×594, #152×593, #236×592 |
| 68 | #249×596, #154×596, #169×594, #226×594 |
| 69 | #134×595, #162×592, #72×592, #229×592 |
| 70 | #34×599, #204×597, #104×593, #55×591 |
| 71 | #221×598, #65×598, #177×594, #26×593 |
| 72 | #28×595, #79×593, #82×593, #77×592 |
| 73 | #188×608, #170×593, #6×592, #57×591 |
| 74 | #13×596, #246×595, #88×594, #85×592 |
| 75 | #227×596, #242×593, #234×592, #107×592 |
| 76 | #112×596, #18×593, #199×592, #105×592 |
| 77 | #107×14, #84×8, #20×7, #55×6 |

## Top experts by language and layer
### Language: `en`
| layer | top experts (id×count) |
|---|---|
| 3 | #132×591, #78×590, #75×590, #47×589 |
| 40 | #146×609, #11×604, #46×599, #70×595 |
| 77 | #107×14, #84×8, #20×7, #55×6 |

## Router entropy
| group | mean entropy (bits) |
|---|---|
| task `coding` | 2.9376 |
| lang `en` | 2.9376 |

## Task overlap (Jaccard of pooled top-N experts)
| task pair | jaccard |
|---|---|

## Language overlap (Jaccard of pooled top-N experts)
| language pair | jaccard |
|---|---|

## Expert specialization (fraction of a task's top-N unique to it)
| task | unique top-N | fraction unique |
|---|---|---|
| coding | 4 | 1.0 |

## Prefill vs generation
| phase | top experts |
|---|---|
| prefill | #29×43357, #71×43356, #121×43356, #116×43356 |
| generation | #104×108, #188×107, #177×107, #87×95 |

Jaccard (top-N) between phases: **0.0**

## Tokenization stats per language (from metadata)
| language | runs | mean prompt tokens |
|---|---|---|---|
| en | 1 | 18745.0 |

## Bounded activation summaries (Phase 4)

- Activation summary records: **563070** across **20** (task, layer, tensor) groups

| task | layer | tensor_stem | topk | n_channels | n_tokens | mean L2 | mean mean | mean std | mean max_abs | top channels |
|---|---|---|---|---|---|---|---|---|---|---|
| coding | 0 | `kv_cmpr` | 20 | 512 | 750760 | 3.746 | 0.0016 | 0.1655 | 1.8795 | #280, #453, #113, #46 |
| coding | 8 | `kv_cmpr` | 20 | 512 | 750760 | 4.7361 | 0.0139 | 0.2087 | 1.9447 | #272, #260, #438, #155 |
| coding | 16 | `kv_cmpr` | 20 | 512 | 750760 | 8.6421 | 0.0114 | 0.3817 | 2.8833 | #411, #245, #231, #281 |
| coding | 24 | `kv_cmpr` | 20 | 512 | 750760 | 8.6112 | -0.0101 | 0.3802 | 2.5844 | #443, #312, #311, #441 |
| coding | 32 | `kv_cmpr` | 20 | 512 | 750760 | 8.367 | -0.0066 | 0.3695 | 2.1779 | #33, #120, #312, #292 |
| coding | 40 | `kv_cmpr` | 20 | 512 | 750760 | 10.1151 | 0.0022 | 0.4468 | 3.5065 | #372, #438, #52, #121 |
| coding | 48 | `kv_cmpr` | 20 | 512 | 750760 | 12.7036 | 0.0212 | 0.5606 | 3.4598 | #441, #369, #254, #182 |
| coding | 56 | `kv_cmpr` | 20 | 512 | 750760 | 11.6412 | 0.0067 | 0.514 | 2.8947 | #445, #185, #187, #249 |
| coding | 64 | `kv_cmpr` | 20 | 512 | 750760 | 13.7618 | 0.0001 | 0.6078 | 3.0485 | #189, #294, #506, #291 |
| coding | 72 | `kv_cmpr` | 20 | 512 | 750760 | 15.2157 | -0.0302 | 0.6713 | 3.9312 | #335, #447, #381, #26 |
| coding | 0 | `q_nope_absorbed` | 20 | 512 | 375380 | 0.2487 | -0.0 | 0.011 | 0.0414 | #17, #269, #440, #97 |
| coding | 8 | `q_nope_absorbed` | 20 | 512 | 375380 | 55.2794 | 0.1031 | 2.4403 | 8.3366 | #322, #275, #465, #449 |
| coding | 16 | `q_nope_absorbed` | 20 | 512 | 375380 | 35.6816 | 0.0177 | 1.5765 | 5.515 | #408, #218, #153, #113 |
| coding | 24 | `q_nope_absorbed` | 20 | 512 | 375380 | 19.4218 | -0.0807 | 0.8543 | 2.7115 | #81, #417, #41, #454 |
| coding | 32 | `q_nope_absorbed` | 20 | 512 | 375380 | 21.0977 | -0.0032 | 0.9319 | 3.9261 | #259, #23, #264, #469 |
| coding | 40 | `q_nope_absorbed` | 20 | 512 | 375380 | 16.212 | -0.0232 | 0.7158 | 2.3054 | #46, #80, #308, #472 |
| coding | 48 | `q_nope_absorbed` | 20 | 512 | 375380 | 24.0615 | -0.0016 | 1.0622 | 3.5006 | #310, #457, #510, #393 |
| coding | 56 | `q_nope_absorbed` | 20 | 512 | 375380 | 12.2422 | 0.0089 | 0.5408 | 1.7813 | #224, #40, #266, #132 |
| coding | 64 | `q_nope_absorbed` | 20 | 512 | 375380 | 19.3848 | 0.0469 | 0.8548 | 2.7705 | #294, #300, #390, #326 |
| coding | 72 | `q_nope_absorbed` | 20 | 512 | 375380 | 13.7915 | 0.0167 | 0.6088 | 1.994 | #510, #369, #123, #404 |

_Top channels ranked by frequency-of-appearance in any token's top-K activation list (not by magnitude). n_tokens is the total (token, channel) pair count contributing to that group. Stat columns are means across per-token values._

## MLA retrieval analysis (Phase 3 / Story 5 re-scoped)

Approximate MLA-latent-attention retrieval from `q_nope_absorbed` queries and `kv_cmpr` keys (captured via `--trace-activations q_nope_absorbed,kv_cmpr`). For each generation-step query (token, layer), scored every earlier prefill position by signed top-K channel overlap (Σ over shared channels of q·c × k·c), ranked top-N positions by descending score.

- Query records seen: **187690** (generation-step only — prefill queries aren't retrieval)
- Key records seen: **375380** (prefill + gen keys)
- (query, layer) pairs scored: **240**
- Sentinel position range: **`50..60`** (tokenized)

### Distance buckets (all retrieved positions)

| bucket | count | fraction | threshold (vs prompt_len) |
|---|---|---|---|
| recent | 47 | 0.020 | ≤5% (or ≤64) |
| medium | 0 | 0.000 | 5%–30% |
| far | 0 | 0.000 | 30%–70% |
| very_far | 2353 | 0.980 | >70% (start of prompt) |
| future | 0 | 0.000 | future (retrieved_pos ≥ query_pos) |

### Sentinel section retrieval

- Sentinel position range (tokenized): `50..60` (inclusive)
- Sentinel hits: **42** / 240 (query, layer) pairs
- Hit rate: **0.175** (fraction of retrieval analyses whose top-N included ≥1 sentinel token position)

### Sample retrieval results (latest 20)

| run_id | layer | query_token | top retrieved positions (pos@score, shared) |
|---|---|---|---|
| `retr_longc` | 0 | 18767 | 10@0.005(6), 26@0.005(6), 194@0.004(7), 360@0.004(7), 7@0.004(5) |
| `retr_longc` | 8 | 18767 | 96@2.387(3), 422@1.714(2), 294@1.674(2), 279@1.649(2), 457@1.572(4) |
| `retr_longc` | 16 | 18767 | 255@2.001(2), 79@1.847(2), 281@1.753(2), 268@1.732(2), 267@1.725(2) |
| `retr_longc` | 24 | 18767 | 451@5.383(4), 509@3.381(3), 494@2.852(2), 490@2.678(2), 306@2.667(2) |
| `retr_longc` | 32 | 18767 | 189@3.762(2), 355@3.738(2), 271@3.709(2), 190@3.624(2), 356@3.598(2) |
| `retr_longc` | 40 | 18767 | 205@5.020(3), 446@4.090(2), 337@3.684(2), 297@3.459(2), 20@3.364(2) |
| `retr_longc` | 48 | 18767 | 153@8.215(3), 196@6.835(2), 362@5.921(2), 345@5.841(2), 295@5.496(2) |
| `retr_longc` | 56 | 18767 | 18765@5.486(2), 455@5.386(2), 492@5.065(2), 437@4.626(2), 50@4.566(2) |
| `retr_longc` | 64 | 18767 | 357@9.732(4), 305@9.001(2), 354@7.628(2), 304@7.430(2), 252@7.252(2) |
| `retr_longc` | 72 | 18767 | 169@5.949(2), 400@5.858(2), 322@5.399(2), 351@5.215(2), 444@4.891(2) |
| `retr_longc` | 0 | 18768 | 249@0.001(7), 74@0.001(8), 135@0.001(5), 106@0.001(7), 448@0.001(7) |
| `retr_longc` | 8 | 18768 | 457@3.018(4), 96@1.698(2), 241@1.563(2), 398@1.509(2), 438@1.429(2) |
| `retr_longc` | 16 | 18768 | 281@2.674(4), 267@2.528(3), 268@2.342(3), 485@2.222(3), 255@1.727(2) |
| `retr_longc` | 24 | 18768 | 314@5.420(5), 148@4.080(4), 509@3.931(3), 387@3.496(3), 221@3.445(3) |
| `retr_longc` | 32 | 18768 | 292@4.712(3), 18764@4.142(3), 27@4.109(3), 30@3.554(2), 52@2.845(2) |
| `retr_longc` | 40 | 18768 | 22@5.219(3), 18759@5.116(3), 335@4.348(2), 70@3.445(2), 157@3.190(2) |
| `retr_longc` | 48 | 18768 | 18759@8.610(3), 171@8.457(3), 284@6.630(2), 510@6.402(2), 483@6.270(2) |
| `retr_longc` | 56 | 18768 | 57@7.770(5), 238@4.470(2), 302@4.381(2), 492@4.125(2), 151@3.706(1) |
| `retr_longc` | 64 | 18768 | 311@13.383(3), 351@9.242(2), 476@8.889(2), 79@8.404(2), 153@8.192(2) |
| `retr_longc` | 72 | 18768 | 407@9.385(3), 415@8.613(3), 511@7.070(3), 144@6.679(2), 400@6.579(2) |

_Approximation only — full activation vectors are not stored; this captures top-K channels per (token, layer). Signed overlap on top-K is a defensible lower-bound interpretability signal of which earlier positions share dominant latent dimensions with the current query._ _Original DSA indexer premise rejected on 2026-06-20 — see `GLM52_SESSION_MEMORY.md` for the full forensic record._

## Runs
| trace | records | language | task | gen/s | dropped | sampled |
|---|---|---|---|---|---|---|
| glm52-coding-en-retr_longctx-20260620-233202.jsonl | 1952001 | en | coding | 0.896 | 0 | 0 |

MLA retrieval-pattern tracing (Phase 3 / Story 5 re-scoped) is **on** — the report above includes the MLA retrieval analysis section. Activation summaries (Phase 4) are also present. See `GLM52_TRACE_PLAN.md` for the re-scope rationale.
