"""Precise: feed the chat-templated prompt, and find per-layer attention output std per layer.
Tests whether attention or FFN is exploding.
"""
import sys, time
import numpy as np
import mlx.core as mx
from mlx_lm import load

OUT = "/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-shortgpt-pruned-4bit-mlx"
print(f"Loading {OUT}", flush=True)
model, tokenizer = load(OUT)
print(f"Model: {type(model).__module__}.{type(model).__name__}\n", flush=True)

messages = [{"role": "user", "content": "What is 2+2?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
    chat_template_kwargs={"enable_thinking": False, "reasoning_effort": None})
ids = mx.array(tokenizer.encode(prompt))[None]
print(f"Tokens: {ids.shape[1]}\n")

emb = model.model.embed_tokens(ids)
mx.eval(emb)
h = emb
n_layers = len(model.model.layers)
print(f"embed std={np.array(h).astype(np.float64).std():.5f}")
print("L    | input     attn-nm   attn-out  pst-attn   ffn-nm    ffn-out   pst-ffn | max|attn_out| max|ffn_out|")
for i, layer in enumerate(model.model.layers):
    if i >= 30:  # 0..29 detailed
        break
    mx.eval(h)
    in_std = np.array(h).astype(np.float64).std()
    r = layer.self_attn(layer.input_layernorm(h), None, None)
    mx.eval(r)
    a_std = np.array(r).astype(np.float64).std()
    a_max = np.abs(np.array(r).astype(np.float64)).max()
    h_post_attn = h + r
    mx.eval(h_post_attn)
    pa_std = np.array(h_post_attn).astype(np.float64).std()
    r2 = layer.mlp(layer.post_attention_layernorm(h_post_attn))
    mx.eval(r2)
    f_std = np.array(r2).astype(np.float64).std()
    f_max = np.abs(np.array(r2).astype(np.float64)).max()
    h_post_ffn = h_post_attn + r2
    mx.eval(h_post_ffn)
    pf_std = np.array(h_post_ffn).astype(np.float64).std()
    print(f"{i:3d}  | {in_std:.5f}  {a_std:.5f}  {a_std:.5f}  {pa_std:.5f}  {f_std:.5f}  {f_std:.5f}  {pf_std:.5f} | {a_max:.4f}  {f_max:.4f}")
    h = h_post_ffn
