"""
Strip the GLM-5.2 MTP (Multi-Token-Prediction) block — `model.layers.78.*` —
from a JANGTQ bundle.

WHY
---
GLM-5.2 ships num_hidden_layers=78 (main layers 0–77) PLUS one MTP block stored
at layer index 78 (num_nextn_predict_layers=1; telltale tensors eh_proj/enorm/
hnorm/shared_head). The stock `mlx_lm` glm_moe_dsa model class only instantiates
the 78 main layers, so layer-78's routed-expert TQ groups have no target module.
The JANGTQ loader hard-fails rather than silently route them through fp16.

MTP is a training/speculative-decode auxiliary; standard autoregressive
inference never uses it. Removing layer-78 tensors makes the bundle load.

WHAT
----
  * Rewrites shards that mix layer-78 with real tensors (drops only the 78.* keys).
  * Deletes shards that are purely layer-78.
  * Rewrites model.safetensors.index.json without the dropped keys.
  * Leaves config.json's num_hidden_layers untouched (mlx_lm builds 0..77; the
    MTP block at 78 simply isn't present, which is exactly what the runtime wants).

USAGE
    python3 strip_mtp_layer.py <bundle_dir> [--layer 78]
"""
import sys
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def shard_keys(sf_path):
    # Use the safetensors library (already imported at module load) instead of
    # re-implementing the binary header parse. Same result, one source of truth.
    with safe_open(str(sf_path), framework="numpy") as f:
        return list(f.keys())


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    BUNDLE = Path(sys.argv[1])
    layer = 78
    if "--layer" in sys.argv:
        layer = int(sys.argv[sys.argv.index("--layer") + 1])
    PREFIX = f"model.layers.{layer}."

    idx_path = BUNDLE / "model.safetensors.index.json"
    idx = json.load(open(idx_path))
    weight_map = idx["weight_map"]

    drop_keys = [k for k in weight_map if k.startswith(PREFIX)]
    if not drop_keys:
        print(f"No keys with prefix {PREFIX!r} — nothing to strip.")
        return
    affected = sorted(set(weight_map[k] for k in drop_keys))
    print(f"Stripping {len(drop_keys)} '{PREFIX}*' keys from {len(affected)} shards")

    for shard_name in affected:
        sf = BUNDLE / shard_name
        keys = shard_keys(sf)
        keep = [k for k in keys if not k.startswith(PREFIX)]
        drop = [k for k in keys if k.startswith(PREFIX)]
        if not keep:
            # Pure-MTP shard → delete outright.
            sf.unlink()
            print(f"  {shard_name}: deleted (all {len(drop)} keys were layer-{layer})")
        else:
            # Mixed shard → rewrite with only the kept tensors.
            with safe_open(str(sf), framework="numpy") as f:
                kept = {k: f.get_tensor(k) for k in keep}
            tmp = sf.with_suffix(".safetensors.tmp")
            save_file(kept, str(tmp))
            tmp.replace(sf)
            print(f"  {shard_name}: rewrote, kept {len(keep)}, dropped {len(drop)}")

    # Rewrite index without dropped keys. (Shard filenames are unchanged; we do
    # NOT renumber — the loader globs model-*.safetensors and reads each header,
    # so a gap in numbering from a deleted pure-MTP shard is harmless.)
    new_map = {k: v for k, v in weight_map.items() if not k.startswith(PREFIX)}
    # Drop any references to now-deleted shards (defensive; none should remain).
    present = {p.name for p in BUNDLE.glob("model-*.safetensors")}
    new_map = {k: v for k, v in new_map.items() if v in present}
    idx["weight_map"] = new_map
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)
    print(f"Index rewritten: {len(weight_map)} → {len(new_map)} keys, "
          f"{len(present)} shards present")
    print("Done.")


if __name__ == "__main__":
    main()
