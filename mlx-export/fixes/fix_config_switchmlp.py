"""Add switch_mlp.{gate,up,down}_proj quantization entries to config.json.

mlx-lm's deepseek_v32.sanitize() stacks per-expert tensors
(model.layers.N.mlp.experts.E.{gate,up,down}_proj) into
model.layers.N.mlp.switch_mlp.{gate,up,down}_proj at load time. The
class_predicate that drives nn.quantize() looks up the STACKED module path in
config["quantization"]. Our converter only recorded per-expert entries, so the
stacked path falls through to the default bits=4 — but the experts are stored at
2-bit, producing a shape mismatch (expected (256,2048,768) got (256,2048,384)).

Fix: for every layer that has per-expert quant entries, add a switch_mlp entry
for each of gate/up/down_proj using the experts' bits/group_size/mode.
"""
import sys
import json

CONFIG = sys.argv[1]
with open(CONFIG) as f:
    config = json.load(f)

q = config.get("quantization", {})
qc = config.get("quantization_config", {})

# Collect per-layer expert quant params (assume uniform across experts of a layer)
# key pattern: model.layers.N.mlp.experts.E.{gate,up,down}_proj
added = 0
import re
seen = {}  # (layer, proj) -> rule
for k, v in list(q.items()):
    m = re.match(r"(model\.layers\.(\d+))\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$", k)
    if m and isinstance(v, dict):
        layer_prefix = m.group(1)
        proj = m.group(3)
        seen[(layer_prefix, proj)] = v

for (layer_prefix, proj), rule in seen.items():
    sm_key = f"{layer_prefix}.mlp.switch_mlp.{proj}"
    if sm_key not in q:
        q[sm_key] = rule
        added += 1

config["quantization"] = q
# keep quantization_config in sync (mirror)
config["quantization_config"] = q

with open(CONFIG, "w") as f:
    json.dump(config, f, indent=2)

print(f"Added {added} switch_mlp quant entries. Total quant keys: {len(q)}")
# Show a couple
import itertools
for k in itertools.islice((k for k in q if "switch_mlp" in k), 0, 4):
    print(f"  {k}: {q[k]}")
