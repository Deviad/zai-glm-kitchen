#!/usr/bin/env bash
# Verify local REAP37 MLX model metadata without loading weights into MLX.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/Volumes/Data NVME/GLM-5.2-REAP37-MLX-4bit-indexer-compat}"

python3 - <<PY
import json
from pathlib import Path
root = Path("$MODEL_DIR")
config = json.loads((root / "config.json").read_text())
index = json.loads((root / "model.safetensors.index.json").read_text())
print("model_dir:", root)
print("model_type:", config.get("model_type"))
print("architectures:", config.get("architectures"))
print("n_routed_experts:", config.get("n_routed_experts"))
print("num_experts_per_tok:", config.get("num_experts_per_tok"))
print("num_hidden_layers:", config.get("num_hidden_layers"))
print("num_nextn_predict_layers:", config.get("num_nextn_predict_layers"))
print("quantization:", config.get("quantization") or config.get("quantization_config"))
print("index_total_size_GiB:", round(index.get("metadata", {}).get("total_size", 0) / 1024**3, 2))
print("index_total_parameters_B:", round(index.get("metadata", {}).get("total_parameters", 0) / 1e9, 2))
shards = sorted(root.glob("model-*.safetensors"))
print("safetensor_shards:", len(shards))
print("disk_size:")
PY

du -sh "$MODEL_DIR"
