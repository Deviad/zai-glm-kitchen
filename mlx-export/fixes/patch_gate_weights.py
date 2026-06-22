"""Patch the corrupted mlp.gate.weight (MoE router) tensors in an already-converted
MLX model, using corrected values from the source GGUF.

The original converter scrambled every 2D F32 tensor via reshape+transpose; in
GLM-5.2 the only such tensors are the MoE routers (ffn_gate_inp.weight). This
script re-reads those from the GGUF with the FIXED dequant_tensor and rewrites
only the safetensors shards that contain a gate.weight, leaving the expensive
quantized expert tensors untouched.
"""
import sys
import json
import re
from pathlib import Path
import numpy as np
import mlx.core as mx

sys.path.insert(0, "mlx-export")
import gguf
from gguf_to_mlx_streaming import dequant_tensor

GGUF_DIR = Path(sys.argv[1])
MLX_DIR = Path(sys.argv[2])

# 1. Read all corrected gate weights from GGUF: gguf name blk.N.ffn_gate_inp.weight
#    -> hf name model.layers.N.mlp.gate.weight
print("Reading corrected router weights from GGUF...", flush=True)
corrected = {}  # hf_name -> mx.array (fp16)
shards = sorted(GGUF_DIR.glob("*.gguf"))
for sh in shards:
    r = gguf.GGUFReader(str(sh))
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.ffn_gate_inp\.weight$", t.name)
        if m:
            hf = f"model.layers.{m.group(1)}.mlp.gate.weight"
            arr = dequant_tensor(t)  # FIXED: (256, 6144) correct layout
            corrected[hf] = mx.array(np.array(arr, dtype=np.float16))
            print(f"  {t.name} -> {hf}  shape={arr.shape}", flush=True)
    del r
print(f"Total corrected routers: {len(corrected)}", flush=True)

# 2. Find which shards contain gate.weight tensors
idx = json.load(open(MLX_DIR / "model.safetensors.index.json"))
wm = idx["weight_map"]
gate_keys = [k for k in wm if k.endswith("mlp.gate.weight")]
shards_to_patch = sorted(set(wm[k] for k in gate_keys))
print(f"Shards to patch: {len(shards_to_patch)}", flush=True)

# Sanity: every gate key must have a correction
missing = [k for k in gate_keys if k not in corrected]
if missing:
    print(f"ERROR: {len(missing)} gate keys missing from GGUF: {missing[:5]}", flush=True)
    sys.exit(1)

# 3. For each shard, load, replace the gate.weight tensors, save back
patched_count = 0
for shard_name in shards_to_patch:
    shard_path = MLX_DIR / shard_name
    weights = mx.load(str(shard_path))
    n_in_shard = 0
    for k in list(weights.keys()):
        if k.endswith("mlp.gate.weight") and k in corrected:
            old = weights[k]
            new = corrected[k]
            assert old.shape == new.shape, f"shape mismatch {k}: {old.shape} vs {new.shape}"
            weights[k] = new.astype(old.dtype)
            n_in_shard += 1
            patched_count += 1
    if n_in_shard:
        mx.save_safetensors(str(shard_path), weights, metadata={"format": "mlx"})
        print(f"  patched {shard_name}: {n_in_shard} routers", flush=True)
    del weights

print(f"DONE. Patched {patched_count} router tensors across {len(shards_to_patch)} shards.", flush=True)
