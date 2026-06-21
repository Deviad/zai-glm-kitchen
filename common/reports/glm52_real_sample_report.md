# GLM-5.2 MoE Expert-Routing Trace Report

## Reproducibility provenance

All runs in this report traced the same model. Re-run with:

```text
/Users/spotted/projects/llama.cpp/build-metal/bin/llama-trace-moe --model /Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/GLM-5.2-mixed-00001-of-00009.gguf -ngl 999 --ctx-size 4096 --predict 12 --temp 0.0 --prompt Implement a non-recursive merge sort in Python.
```

- Model: `GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest`
- prompt_sha256 (this run): `afe76262a9a629a8b08277b0ee65f75e2d5cec3b01394193fe5a7bd06c3158ae`
- model_sha256_prefix (first 1 MiB): `78a23335f717461a`
- model_total_size_bytes: **249186991232** (232.1 GiB)
- Run window: `2026-06-20T19:33:56Z` → `2026-06-20T19:35:19Z` (UTC)
- Speed: **0.92 gen tok/s**, 6.23 prefill tok/s (1 gen tokens / 10 prompt tokens)
- n_expert_total: **256** (total routed experts per MoE layer)
- Sample run_id: `story8_smoke-en-1781984116`

- Records traced: **24** across **1** run(s)
- Layers seen: **12** (3..14)
- Experts per routing event: up to **8** of **256** total
- Tasks: `coding`
- Languages: `en`

## Top experts by task and layer
### Task: `coding`
| layer | top experts (id×count) |
|---|---|
| 3 | #28×2, #73×2, #197×2, #250×1 |
| 4 | #209×1, #225×1, #185×1, #101×1 |
| 5 | #92×1, #125×1, #80×1, #11×1 |
| 6 | #106×1, #192×1, #36×1, #166×1 |
| 7 | #52×1, #108×1, #210×1, #10×1 |
| 8 | #39×1, #23×1, #31×1, #14×1 |
| 9 | #122×1, #119×1, #194×1, #222×1 |
| 10 | #223×1, #24×1, #150×1, #249×1 |
| 11 | #173×1, #27×1, #44×1, #79×1 |
| 12 | #145×1, #155×1, #220×1, #138×1 |
| 13 | #57×1, #85×1, #183×1, #200×1 |
| 14 | #232×1, #124×1, #7×1, #135×1 |

## Top experts by language and layer
### Language: `en`
| layer | top experts (id×count) |
|---|---|
| 3 | #28×2, #73×2, #197×2, #250×1 |
| 9 | #122×1, #119×1, #194×1, #222×1 |
| 14 | #232×1, #124×1, #7×1, #135×1 |

## Router entropy
| group | mean entropy (bits) |
|---|---|
| task `coding` | 2.8826 |
| lang `en` | 2.8826 |

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
| prefill | #177×2, #5×2, #185×2, #114×2 |
| generation | #80×3, #145×3, #28×2, #197×2 |

Jaccard (top-N) between phases: **0.0**

## Tokenization stats per language (from metadata)
| language | runs | mean prompt tokens |
|---|---|---|
| en | 1 | 10.0 |

## Runs
| trace | records | language | task | gen/s | dropped | sampled |
|---|---|---|---|---|---|---|
| glm52-coding-en-real-sample.jsonl | 24 | en | coding | 0.9171 | 0 | 0 |

DSA long-context retrieval tracing (Phase 3) is **not yet implemented** in this report; see `GLM52_TRACE_PLAN.md`. Activation summaries (Phase 4) are disabled by default and require explicit flags.
