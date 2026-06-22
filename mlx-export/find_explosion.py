"""Find the exact layer where the activation explodes in GLM-5.2 MLX forward.
"""
import sys
import numpy as np
import mlx.core as mx
from mlx_lm import load

MODEL = "/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-shortgpt-pruned-4bit-mlx"

print(f"Loading: {MODEL}", flush=True)
model, tokenizer = load(MODEL)
print(f"Model: {type(model).__module__}.{type(model).__name__}\n", flush=True)

# Use raw prompt (no chat template) to rule that out
prompt_text = "What is 2+2?"
input_ids = mx.array(tokenizer.encode(prompt_text))[None]
print(f"Tokens: {input_ids.shape[1]} | raw prompt: {repr(prompt_text)}\n")

# Get embeddings
h = model.model.embed_tokens(input_ids)
mx.eval(h)
arr0 = np.array(h)
print(f"embed: mean={arr0.mean():.4f} std={arr0.std():.4f} min={arr0.min():.4f} max={arr0.max():.4f}")

# Scan ALL layers, find where std explodes
prev_std = float(arr0.std())
n_layers = len(model.model.layers)
for i, layer in enumerate(model.model.layers):
    h = layer(h, None)
    mx.eval(h)
    arr = np.array(h).astype(np.float64)
    s = arr.std()
    # report big jumps
    marker = ""
    if s > prev_std * 1.5:
        marker = " ◄≈1.5x jump"
    if s > 1.0 and prev_std < 1.0:
        marker += " ← O(1) reached"
    if s > 1e6 or np.isinf(s) or np.isnan(arr).any():
        marker += " 🔥 EXPLODED"
    print(f"  layer {i:3d}: std={s:.4f} mean={arr.mean():.4f} min={arr.min():.4f} max={arr.max():.4f}{marker}")
    prev_std = s
    if np.isinf(s) or np.isnan(arr).any():
        print(f"  (stopping at layer {i})")
        break

print(f"\n=== Layer-by-layer attn+ffn stats for first 6 layers ===")
h = model.model.embed_tokens(input_ids)
for i, layer in enumerate(model.model.layers):
    if i >= 6:
        break
    mx.eval(h)
    in_arr = np.array(h).astype(np.float64)
    # Apply components
    resid = h
    h_n = layer.input_layernorm(h)
    mx.eval(h_n)
    attn = layer.self_attn(h_n, None)
    mx.eval(attn)
    post_attn = resid + attn
    mx.eval(post_attn)
    h_n2 = layer.post_attention_layernorm(post_attn)
    mx.eval(h_n2)
    ffn = layer.mlp(h_n2)
    mx.eval(ffn)
    post_ffn = resid + ffn
    mx.eval(post_ffn)
    a_in = in_arr.std()
    a_attn_n = np.array(h_n).astype(np.float64).std()
    a_attn = np.array(attn).astype(np.float64)
    a_pa = np.array(post_attn).astype(np.float64)
    a_ffn_n_arr = np.array(h_n2).astype(np.float64)
    a_ff_arr = np.array(ffn).astype(np.float64)
    a_pf_arr = np.array(post_ffn).astype(np.float64)
    print(f"  L{i}: in={a_in.std():.4f}  attn-norm={a_attn_n:.4f}  attn-out={a_attn.std():.4f}  postattn={a_pa.std():.4f}  ffn-norm={a_ffn_n_arr.std():.4f}  ffn-out={a_ff_arr.std():.4f}  postffn={a_pf_arr.std():.4f}")
    h = post_ffn
