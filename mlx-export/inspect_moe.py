"""Inspect MoE block at layer 4 and 7 — what's the output of each component?"""
import sys
import numpy as np
import mlx.core as mx
from mlx_lm import load

OUT = "/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-shortgpt-pruned-4bit-mlx"
model, tokenizer = load(OUT)

messages = [{"role": "user", "content": "What is 2+2?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
    chat_template_kwargs={"enable_thinking": False, "reasoning_effort": None})
ids = mx.array(tokenizer.encode(prompt))[None]

# Run through layers 0..3, then the MoE layer at 4 — but log internal MoE values
h = model.model.embed_tokens(ids)
mx.eval(h)

for i in range(7):
    h = model.model.layers[i](h, None, None)
    mx.eval(h)

# Now examine layer 7 MoE in detail (first one that explodes)
layer4 = model.model.layers[7]
moe = layer4.mlp
print(f"=== Layer 4 MLP type: {type(moe).__name__} ===")

# Apply input layer norm and pre-attn
norm_input = layer4.input_layernorm(h)
mx.eval(norm_input)
attn_out = layer4.self_attn(norm_input, None, None)
mx.eval(attn_out)
h_attn = h + attn_out
mx.eval(h_attn)
norm_h = layer4.post_attention_layernorm(h_attn)
mx.eval(norm_h)
print(f"pre-mlp input std: {np.array(norm_h).astype(np.float64).std():.5f}")

# MoE forward (manual)
gates_to_compute = norm_h
inds, scores = moe.gate(gates_to_compute)
mx.eval(inds); mx.eval(scores)
print(f"gate inds shape: {inds.shape}, scores shape: {scores.shape}")
print(f"inds sample: {np.array(inds).tolist()}")
print(f"scores sample: {np.array(scores).tolist()}")
print(f"scores sum (should be ~routed_scaling_factor if normalized): per-token sums = {np.array(scores.sum(-1)).tolist()}")

# SwitchGLU output
y = moe.switch_mlp(gates_to_compute, inds)
mx.eval(y)
print(f"switch_mlp out shape: {y.shape}, std: {np.array(y).astype(np.float64).std():.5f}, max: {np.abs(np.array(y).astype(np.float64)).max():.4f}")
# Then weighted sum
y_weighted = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
mx.eval(y_weighted)
print(f"weighted y: std: {np.array(y_weighted).astype(np.float64).std():.5f}, max: {np.abs(np.array(y_weighted).astype(np.float64)).max():.4f}")

# Shared experts
sh_out = moe.shared_experts(norm_h)
mx.eval(sh_out)
print(f"shared expert out: std: {np.array(sh_out).astype(np.float64).std():.5f}, max: {np.abs(np.array(sh_out).astype(np.float64)).max():.4f}")

# Total
total = y_weighted + sh_out
mx.eval(total)
print(f"total MoE out: std: {np.array(total).astype(np.float64).std():.5f}")
