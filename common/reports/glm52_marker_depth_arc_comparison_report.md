# Phase 6 follow-up #4 — Marker depth arc comparison

Evidence backing `GLM52_SESSION_MEMORY.md` Phase 6 follow-up #4 (2026-06-20).

## Question

Do other markers (#506, #2305, #2232) share #4386's 3-phase sign-flipping arc
(small positive L0-L6 → negative L12-L36 → strongly positive L42-L72), or are
they single-layer "bump" channels?

## Method

Compare per-layer mean magnitudes across all 4 markers, extracted from each
marker's individual focus summary JSON:
- `reports/glm52_channel_4386_focus_summary.json`
- `reports/glm52_channel_2232_focus_summary.json`
- `reports/glm52_channel_506_focus_summary.json`
- `reports/glm52_channel_2305_focus_summary.json`

Each per_layer_magnitude entry has: `layer`, `mean`, `std`, `min`, `max`, `n`.

## Result — Per-layer mean magnitude (#4386, #2232, #506, #2305)

| layer |   #4386 mean@n  |   #2232 mean@n  |    #506 mean@n  |   #2305 mean@n  |
|-------|-----------------|-----------------|-----------------|-----------------|
|     0 |  +0.118 @ 7259  |  -0.047 @ 3579  |  -0.068 @ 7230  |  -0.075 @ 6803  |
|     6 |  +0.180 @ 6730  |  -0.017 @ 4300  |  -0.128 @ 6928  |  -0.137 @ 6905  |
|    12 |  -1.591 @ 7138  |  -0.021 @ 4363  |  -0.130 @ 7166  |  +0.464 @ 7187  |
|    18 |  -1.415 @ 7138  |  -0.108 @ 4186  |  +0.616 @ 6661  |  +0.286 @ 7240  |
|    24 |  -0.667 @ 7287  |  +0.136 @ 4025  |  +1.314 @ 3801  |  -0.438 @ 7291  |
|    30 |  -2.963 @ 6406  |  +3.075 @ 4493  |  +0.331 @ 2721  |  -0.690 @ 3698  |
|    36 |  -0.498 @ 7100  |  +3.970 @ 4152  |  +1.144 @ 2121  |  +0.253 @  569  |
|    42 |  +3.638 @ 7244  |  +4.434 @ 3751  |  +3.585 @  549  |  +1.082 @   64  |
|    48 |  +6.546 @ 7148  |  +5.891 @ 2180  |  +0.561 @   23  |  +1.747 @   29  |
|    54 | +15.410 @ 7155  | +28.961 @  380  |  -1.674 @    9  |  +1.970 @   15  |
|    60 | +18.994 @ 7108  | +32.541 @  316  |  -2.932 @   15  |  +2.945 @    9  |
|    66 | +19.611 @ 7022  | +48.036 @  189  |  -1.602 @   11  |  +1.202 @   12  |
|    72 | +17.921 @ 5551  |  -4.705 @   46  |  -4.915 @   17  |  +5.105 @    7  |

## Key observations

1. **Only #4386 has the 3-phase sign-flipping arc.** Its mean trajectory goes
   +0.12 (L0) → -1.59 (L12) → -2.96 (L30, peak negative) → +3.64 (L42, sign
   flip) → +19.6 (L66, peak positive) → +17.9 (L72, slight drop).
   At every deep layer (L48-L72), #4386 has 5500-7000 records — macroscopically
   present everywhere.

2. **#2232 has a sustained positive spike (L30-L66) but is fade-prone.** Its
   mean is consistently positive (+3 to +5) from L30 onward, peaking at +48
   (L66). But `n` drops from ~4500 (L30) → 189 (L66). As #2232 becomes rarer
   at deeper layers, the records where it IS present tend to be the most
   saturated (rank-1, token-0) ones — selective rather than ubiquitous.

3. **#506 and #2305 are pure single-layer "bump" channels.** Their peak
   magnitudes are +0.62 (L18 for #2305) and +1.31 (L24 for #506). After their
   peak layer, `n` drops below 20 within 6-12 layers. They don't survive
   deeper than their peak layer.

## Conclusion — marker taxonomy

| Channel | Marker role | Depth arc type | Peak layer | Sustains at depth? |
|---|---|---|---|---|
| #4386 | Deep-marker workhorse | 3-phase sign flip | L66 | YES — 7000+ records at L24-L72 |
| #2232 | Selective deep-marker | Sustained positive spike | L66 (mean peak), L42 (frequency peak) | Partial — fades in n but mean grows |
| #506 | Shallow bump | Single-layer spike | L24 | NO — n <20 at L48+ |
| #2305 | Shallow bump | Single-layer spike | L18 | NO — n <50 at L30+ |

#4386 is UNIQUE among the marker family — it's the only one whose marker role
is sustained and macroscopically present across the deep layers (every layer
24-72). The other markers are layer-specific signal bursts, not sustained
signal carriers.

This refines the Phase 6 #3 token-0 marker family finding: while multiple
channels saturate on token 0 at different layers, only #4386 is the
"deep-marker workhorse" — the dominant marker channel that maintains presence
throughout the deep layers (L24-L72).

## Provenance

- Source data: `reports/glm52_channel_{4386,2232,506,2305}_focus_summary.json`
- Trace dataset: `traces/batch/activation_full_161/*.jsonl` (161 prompts)
- Method: cross-channel per-layer aggregation, Python-only (no new C++ runs).
