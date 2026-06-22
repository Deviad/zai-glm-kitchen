"""Verify the actual shape of `k` produced by embed_q on real model."""
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention

OUT = "/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-shortgpt-pruned-4bit-mlx"
model, tokenizer = load(OUT)
messages = [{"role": "user", "content": "What is 2+2?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
    chat_template_kwargs={"enable_thinking": False, "reasoning_effort": None})
ids = mx.array(tokenizer.encode(prompt))[None]
h = model.model.embed_tokens(ids)
attn = model.model.layers[0].self_attn
norm_h = model.model.layers[0].input_layernorm(h)
B, L, D = norm_h.shape
qr = attn.q_a_layernorm(attn.q_a_proj(norm_h))
q = attn.q_b_proj(qr)
q = q.reshape(B, L, attn.num_heads, attn.q_head_dim).transpose(0, 2, 1, 3)
q_nope, q_pe = mx.split(q, [attn.qk_nope_head_dim], axis=-1)
compressed_kv = attn.kv_a_proj_with_mqa(norm_h)
compressed_kv, k_pe = mx.split(compressed_kv, [attn.kv_lora_rank], axis=-1)
k_pe = k_pe.reshape(B, L, 1, attn.qk_rope_head_dim).transpose(0, 2, 1, 3)
kv_latent = attn.kv_a_layernorm(compressed_kv)
kv_latent = mx.expand_dims(kv_latent, axis=1)

print(f"kv_latent shape: {kv_latent.shape}")
print(f"kv_latent[..., None, :, :] shape: {kv_latent[..., None, :, :].shape}")
print()
k = attn.embed_q(kv_latent[..., None, :, :], transpose=False)
v = attn.unembed_out(kv_latent)
print(f"k (embed_q result) shape: {k.shape}")
print(f"v (unembed_out result) shape: {v.shape}")
print()
print(f"q_nope shape: {q_nope.shape}")

# Test scaled_dot_product_attention works at all with these shapes
try:
    print(f"\nAttempting sdpa call...")
    out = scaled_dot_product_attention(q_nope, k, v, cache=None, scale=attn.scale, mask=None)
    mx.eval(out)
    print(f"  out shape: {out.shape}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
