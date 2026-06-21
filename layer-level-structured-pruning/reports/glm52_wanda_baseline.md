# GLM-5.2 Wanda pruning — baseline measurement

**Created:** 2026-06-21
**Status:** Pre-prune baseline verified
**Reference copy:** `/Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-wanda-IQ2S-experts-IQ4NL-rest/`

## Source

Real file copy (~227 GB, 9 shards) of the MTP-pruned model. No symlinks.
Phase 7b verified the pruned-MTP model produces **byte-identical routing +
activation traces** to the original mixed baseline (0 real diffs across
6815 records at float tol 1e-9).

## Disk layout

- `GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest/` (original) — **preserved, read-only**
- `GLM-5.2-pruned-IQ2S-experts-IQ4NL-rest/` (blk.78.* tensors dropped) — **preserved, read-only**
- `GLM-5.2-wanda-IQ2S-experts-IQ4NL-rest/` — **working copy for Wanda pruning**
- `UD-IQ4_NL/` — unrelated

## Verified post-copy load (2026-06-21)

`llama-cli -p "Hello. Reply with one word." -n 4`:

```
[ Prompt: 40.6 t/s | Generation: 27.2 t/s ]
```

Matches pruned-MTP baseline (40.7 / 23.7) within noise.
