#!/usr/bin/env python3
"""
Load-time patch test v2:
- Keep JANG switch_mlp (experts, 4-bit) — the thing we want to test
- Replace ALL non-expert tensors from the known-good 4-bit source
- Strip orphan nested-quant keys
- Rename language_model.lm_head → lm_head
No files modified — validates the concept.
"""
import json, re, sys, time, traceback
from pathlib import Path
from collections import defaultdict

import mlx.core as mx
from safetensors import safe_open
import numpy as np

JANG = Path("/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-JANG_4L")
SRC  = Path("/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-shortgpt-pruned-4bit-mlx")

# ── 1. Pre-load ALL source non-expert tensors ────────────────────────
print("Loading source non-expert tensors …", flush=True)
t0 = time.time()
src_idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]

# Group needed keys by shard (skip individual expert keys)
shard_keys = defaultdict(list)
for k, shard_name in src_idx.items():
    if ".mlp.experts." in k:
        continue  # skip source experts — JANG provides pre-stacked switch_mlp
    shard_keys[shard_name].append(k)

src_all = {}
total_keys = sum(len(v) for v in shard_keys.values())
print(f"  {total_keys} non-expert keys across {len(shard_keys)} shards")
for i, (shard_name, keys) in enumerate(sorted(shard_keys.items())):
    with safe_open(str(SRC / shard_name), framework="numpy") as f:
        for k in keys:
            src_all[k] = mx.array(f.get_tensor(k))
    if (i + 1) % 10 == 0:
        print(f"    shard {i+1}/{len(shard_keys)} …", flush=True)
print(f"  done in {time.time()-t0:.1f}s  ({sum(v.nbytes for v in src_all.values())/1e9:.2f} GB)")

# ── 2. Orphan detection ───────────────────────────────────────────────
orphan_re = re.compile(r'\.(scales|biases)\.(weight|scales|biases)$')
def is_orphan(key):
    return bool(orphan_re.search(key))

# ── 3. Monkey-patch mx.load ───────────────────────────────────────────
original_load = mx.load
stats = {"kept_jang": 0, "from_source": 0, "orphans": 0, "dropped_scales": 0, "renamed": 0}

def patched_load(path, *args, **kwargs):
    data = original_load(path, *args, **kwargs)
    result = {}
    for k, v in data.items():
        # Skip orphans
        if is_orphan(k):
            stats["orphans"] += 1
            continue
        # Rename language_model.lm_head → lm_head
        if k.startswith("language_model.lm_head."):
            k = k[len("language_model."):]
            stats["renamed"] += 1
        # Keep JANG experts (switch_mlp) as-is
        if ".switch_mlp." in k:
            result[k] = v
            stats["kept_jang"] += 1
            continue
        # For everything else: use source data if available
        if k in src_all:
            result[k] = src_all[k]
            stats["from_source"] += 1
            continue
        # Source doesn't have this key
        # If it's scales/biases for a tensor source has as UNQUANTIZED → drop it
        # so nn.quantize won't quantize that module (no scales = stays fp16)
        if k.endswith(".scales") or k.endswith(".biases"):
            base = k.rsplit(".", 1)[0]              # e.g. model.embed_tokens
            if base + ".weight" in src_all and base + ".scales" not in src_all:
                stats["dropped_scales"] += 1
                continue
        # Otherwise keep JANG value (might be needed)
        result[k] = v
    return result

mx.load = patched_load

# ── 4. Load model ─────────────────────────────────────────────────────
# Restore mx.load on ANY exit (normal or exception) so the global monkey-patch
# + its src_all closure never leaks past this script. See REMEDIATION_PLAN P2.
print("\nLoading patched model (JANG experts + source everything-else) …", flush=True)
try:
    try:
        from mlx_lm.utils import load_model
        t0 = time.time()
        model, tokenizer = load_model(JANG, lazy=False)
        print(f"  LOAD OK in {time.time()-t0:.1f}s — {type(model).__name__}")
        print(f"  stats: {stats}")
    except Exception as e:
        print(f"\n  LOAD FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"\n  stats before failure: {stats}")
        sys.exit(1)
finally:
    mx.load = original_load

# ── 5. Generate ───────────────────────────────────────────────────────
print("\nGenerating …", flush=True)
import mlx_lm
prompt = "What is 2+2?"
try:
    t0 = time.time()
    response = mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=200, verbose=True
    )
    print(f"\nGeneration took {time.time()-t0:.1f}s")
except Exception as e:
    print(f"\n  GENERATE FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
