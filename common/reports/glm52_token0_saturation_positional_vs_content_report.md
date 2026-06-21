# Phase 6 follow-up #6 — Token-0 saturation: positional pattern, first-token-text-shaped magnitude

Evidence backing `GLM52_SESSION_MEMORY.md` Phase 6 follow-up #6 (2026-06-20).

## Question

Resolve the two competing hypotheses from Phase 6 #1 about #4386's token-0 saturation:

- **Hypothesis A (pure positional encoding marker)**: token-0 saturates because position 0 is
  encoded by #4386. Prediction: every token-0 saturates to ~-226 regardless of what the first
  token IS.

- **Hypothesis B' (residual saturation, content-mediated)**: token-0 saturates because of
  causal-attention dynamics (no prior tokens to attend to), but the magnitude depends on the
  first-token's embedding. Prediction: token-0 saturates universally, but magnitude varies by
  first-token text.

## Method

Akin to a controlled experiment:
1. Sample 7 English prompts with different first tokens (3 share token 'A', 4 have unique tokens).
2. Extract each prompt's first-token text via `llama-tokenize` (verbatim, no BOS).
3. Look up each prompt's L48 token-0 #4386 magnitude from the existing 161-prompt trace data.
4. Compare within-first-token spread (3 prompts sharing 'A') vs across-first-token spread (4 different first-tokens).

The "A" prompts are very different content:
- `physics_01_projectile`: "A projectile is launched at 20 m/s at 30 degrees..."
- `cybersecurity_01_phishing_triage`: "A user reports an email asking them to reset their password..."
- `engineering_01_safety_factor`: "A bracket must support a 500 N load..."

## Result — per-prompt data

```
 first token ID    first token test_id                              L48 #4386 mag
-------------------------------------------------------------------------------------------
           9880         'Find' math_03_eigenvalues                       -226.250
             32            'A' physics_01_projectile                     -225.978
             32            'A' cybersecurity_01_phishing_triage          -225.978
             32            'A' engineering_01_safety_factor              -225.977
            840           'Ex' cs_02_database_transactions               -225.886
           7984        'Write' coding_01_iterative_merge_sort            -224.983
          21142      'Balance' chemistry_01_balancing                    -224.266
```

## Within-first-token spread

Three prompts share first-token 'A' (id=32) with wildly different content:

| Prompt | First-token | Domain | L48 #4386 mag |
|---|---|---|---|
| physics_01_projectile | 'A' | physics | -225.978 |
| cybersecurity_01_phishing_triage | 'A' | cybersecurity | -225.978 |
| engineering_01_safety_factor | 'A' | engineering | -225.977 |
| **Spread** | | | **0.001** |

**Three radically different content domains produce IDENTICAL token-0 saturation
(up to 3 decimal places) when they share the same first-token.**

## Across-first-tokens spread

Four prompts with four different first tokens:

| First token | Prompt | Domain | L48 #4386 mag |
|---|---|---|---|
| 'Find' | math_03_eigenvalues | math | -226.250 |
| 'Ex'  | cs_02_database_transactions | computer science | -225.886 |
| 'Write' | coding_01_iterative_merge_sort | coding | -224.983 |
| 'Balance' | chemistry_01_balancing | chemistry | -224.266 |
| **Spread** | | | **1.984** |

**Different first-tokens produce saturation magnitudes that vary by ~2 (2000x larger than within-token spread).**

## Per-language distribution (all 23 prompts per language at L48 token-0)

| language | n | min | mean | max | median | std |
|---|---|---|---|---|---|---|
| de | 23 | -226.082 | -224.950 | -223.316 | -224.778 | 0.789 |
| en | 23 | -226.250 | -225.731 | -224.266 | -225.977 | 0.572 |
| es | 23 | -226.133 | -225.432 | -223.887 | -225.620 | 0.715 |
| fr | 23 | -226.182 | -225.520 | -223.709 | -225.688 | 0.765 |
| it | 23 | -226.182 | -225.410 | -221.298 | -225.937 | 1.079 |
| pt | 23 | -226.133 | -224.869 | -223.331 | -224.501 | 0.881 |
| zh | 23 | -226.041 | -224.420 | -217.742 | -225.115 | 1.801 |

All 7 languages reach -226 max saturation. CJK (zh) has looser std (1.8 vs 0.6-1.1 for Latin) — single-character CJK tokens encode differently from multi-character Latin words.

## Conclusion

### Hypothesis A (pure positional marker) is refined, not strictly correct

If positional only, all first-tokens would saturate at the same exact value (-226.25). They don't
— they vary by ~2 magnitudes across different first-tokens. So #4386 is not purely positional.

### Hypothesis B' (residual saturation, content-shaped) is the better fit

Token-0 saturation magnitude depends on what the first token IS (likely because each token's
embedding produces slightly different residual stream dynamics). But the QUALITATIVE pattern of
"token 0 uniquely saturates" is universal — every prompt's token 0 saturates regardless of content.

### Narrowest interpretation: #4386 is a "first-token marker that's content-modulated"

It always fires a strong negative signal on token 0 (the qualitative marker property — 100% of
161 prompts saturate negatively between -217 and -226). But the EXACT magnitude is shaped by the
first token's identity (the quantitative modulation — varies by ~2 magnitudes across first-tokens,
but only ~0.001 within same first-token).

### Brought to a tight conclusion

The fact that **three radically different prompts (physics, cybersecurity, engineering) sharing
first-token 'A' produce IDENTICAL saturation (-225.978 to 3 decimal places)** rules out a
strong content-dependence interpretation. The fact that **different first-tokens saturate to
slightly different values (-224 to -226)** rules out pure positional binding. The combined
result is the refined hypothesis: **#4386 fires on position 0 in a content-modulated way**.

This rules out #4386 as a pure positional encoding primitive, consistent with the causal-attention
amplification theory (token 0 has unique causal attention dynamics regardless of content, but its
actualization into the #4386 channel magnitude depends on the first-token's embedding).

## Provenance

- Trace dataset: `traces/batch/activation_full_161/*.jsonl` (161 prompts, all 7 languages)
- Tokenizer: `llama-tokenize` from `/Users/spotted/projects/llama.cpp/build-metal/bin/llama-tokenize`
- Model: mixed GLM-5.2 GGUF (1st shard)
- Method: single Python script reading pre-existing traces + 7 tokenize invocations. No new C++ runs.
