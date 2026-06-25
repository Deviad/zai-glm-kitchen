# JANGTQ_K MLX speed plan — coding-agent (Architect/Reviewer) at ~100K context

Created: 2026-06-24
Owner: GLM-5.2-kitchen
Status: **DONE — MLA prefix-cache patch shipped.** See GLM52_SESSION_MEMORY.md 2026-06-24 entry.

## Outcome (2026-06-24)

The gating measurement (US-A) revealed the prefix cache was **broken for MLA**,
not just untuned. Root-caused two bugs in vendored vMLX `BlockAwarePrefixCache`
and patched both (see `GLM52_SESSION_MEMORY.md` 2026-06-24 entry for the
symptom → cause → fix → verified trace):

1. **Asymmetric k/v pad shape** in CacheList KV rebuild — `mx.concatenate`
   crash when MLA k/v feature dims differ (e.g. 64 vs 512).
2. **All-skip sub-cache abort** — MLA layers whose secondary head group
   accumulated 0 tokens for the prefix stored `("skip",)` in every block,
   which aborted the whole reconstruct and forced a full re-prefill each
   warm turn.

**Verified results (JANGTQ_K MLX, `--use-paged-cache`, patch installed):**
- Short 480-token prompt: cold 7.5 s → warm 1.1 s, correct.
- 53K BLUE-FALCON: cold **1404.7 s** → warm **59.6 s** = **23.6× speedup**,
  BLUE-FALCON-48217 recovered on both turns (correctness gate passed).

**Deployment:** patch in `vendor/vmlx/vmlx_engine/prefix_cache.py` (submodule
`b7da1b8` + two-hunk diff). Installed into bundled vMLX site-packages so daily
`vmlx-serve` runs pick it up without PYTHONPATH; reinstall after vMLX updates.
Both fixes opt-in via `--use-paged-cache`; default runs unaffected.

## Context

The working coherent-at-53K model is the JANGTQ_K MLX bundle (279 GB, ~3.51 bpw),
served by vMLX. Target workload: coding agents (Architect/Reviewer) at ~100K
context — long stable prefix (system + codebase + docs) + varied short
questions, the exact pattern vMLX's prefix cache is built for.

## User stories + acceptance criteria (status)

### US-A: Measure the prefill/decode split at long context — DONE
Split measured: prefill ~39 tok/s (~97% of wall at 53K); decode ~6.9 tok/s;
warm cache was broken (cache-miss every turn) until the patch. BLUE-FALCON
recovered on both cold and warm.

### US-B: Coding-agent access pattern characterization — N/A (skipped, valid)
Prefill is content-independent; BLUE-FALCON is a valid perf proxy. Coding-agent
correctness is the harder gate and passed at 53K.

### US-C: Sweep `prefill_batch_size` — DEFERRED (no longer gating)
Only helps cold prefill, now the minority of wall (warm turns decode-only via
the patch). Not pursued. Reopen if cold-start cost matters.

### US-D: Decide config vs Metal kernel work — DECIDED
Patch the MLA paged-cache reconstruct (done). No Metal kernel work warranted for
the coding-agent workload — warm turns are already decode-only (~60 s at 53K).

## Out of scope

- New Metal kernels (JANGTQ MXTQ shaders / vmlx engine internals).
- KV-cache quantization tuning (auto-disabled for MLA — dead lever).
- GGUF re-quantization (down_proj → IQ4_NL) — separate deferred track.
- MTP self-speculative decode — user-rejected (~10% ceiling on DSA).
