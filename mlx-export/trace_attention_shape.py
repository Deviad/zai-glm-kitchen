"""Inspect the actual shapes & values produced along the real DeepseekV32Attention forward.

Mimics the exact code path from deepseek_v32.py L179-256 (the L>1 prefill path),
capturing intermediate activations. The bug must be in this path because:
  - The model loads correctly (weights match GGUF).
  - llama.cpp on the same GGUF weights produces coherent output.
  - So somewhere in the MLX compute graph, values diverge.
"""
import time
import traceback
import numpy as np
import mlx.core as mx
from mlx_lm import load

OUT = "/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-shortgpt-pruned-4bit-mlx"

print("Loading...", flush=True)
t0 = time.time()
model, tokenizer = load(OUT)
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

# Build prompt and embeddings
messages = [{"role": "user", "content": "What is 2+2?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
    chat_template_kwargs={"enable_thinking": False, "reasoning_effort": None})
ids = mx.array(tokenizer.encode(prompt))[None]
print(f"Prompt tokens: {ids.shape}")

# === Forward through layer 0 attention manually ===
attn = model.model.layers[0].self_attn
ln = model.model.layers[0].input_layernorm
h = model.model.embed_tokens(ids)

print(f"\nembeddings: shape={h.shape} dtype={h.dtype}")
print(f"  std={np.array(h).astype(np.float64).std():.6f}  max(abs)={np.abs(np.array(h).astype(np.float64)).max():.6f}")

# Layer 0 attention forward — mimicking deepseek_v32.py DeepseekV32Attention.__call__
B, L, D = h.shape
norm_h = ln(h)
print(f"\nnorm_h: shape={norm_h.shape}")
print(f"  std={np.array(norm_h).astype(np.float64).std():.6f}  max(abs)={np.abs(np.array(norm_h).astype(np.float64)).max():.6f}")

qr = attn.q_a_layernorm(attn.q_a_proj(norm_h))
print(f"\nqr (q_a_proj+q_a_layernorm): shape={qr.shape}  q_a_layernorm.eps={attn.q_a_layernorm.eps}")
print(f"  std={np.array(qr).astype(np.float64).std():.6f}  max(abs)={np.abs(np.array(qr).astype(np.float64)).max():.6f}")

q = attn.q_b_proj(qr)
print(f"\nq (q_b_proj output): shape={q.shape}")
print(f"  std={np.array(q).astype(np.float64).std():.6f}  max(abs)={np.abs(np.array(q).astype(np.float64)).max():.6f}")

q = q.reshape(B, L, attn.num_heads, attn.q_head_dim).transpose(0, 2, 1, 3)
q_nope, q_pe = mx.split(q, [attn.qk_nope_head_dim], axis=-1)
print(f"\nq_nope: shape={q_nope.shape}")
print(f"  std={np.array(q_nope).astype(np.float64).std():.6f}")
print(f"q_pe: shape={q_pe.shape}")
print(f"  std={np.array(q_pe).astype(np.float64).std():.6f}")

compressed_kv = attn.kv_a_proj_with_mqa(norm_h)
print(f"\ncompressed_kv: shape={compressed_kv.shape}")
print(f"  std={np.array(compressed_kv).astype(np.float64).std():.6f}  max(abs)={np.abs(np.array(compressed_kv).astype(np.float64)).max():.6f}")

compressed_kv_, k_pe = mx.split(compressed_kv, [attn.kv_lora_rank], axis=-1)
k_pe = k_pe.reshape(B, L, 1, attn.qk_rope_head_dim).transpose(0, 2, 1, 3)
print(f"\nk_pe (after reshape): shape={k_pe.shape}")
print(f"  std={np.array(k_pe).astype(np.float64).std():.6f}")

kv_latent_unexpanded = attn.kv_a_layernorm(compressed_kv_)
print(f"\nkv_latent (unexpanded): shape={kv_latent_unexpanded.shape}  kv_a_layernorm.eps={attn.kv_a_layernorm.eps}")
print(f"  std={np.array(kv_latent_unexpanded).astype(np.float64).std():.6f}  max(abs)={np.abs(np.array(kv_latent_unexpanded).astype(np.float64)).max():.6f}")

kv_latent = mx.expand_dims(kv_latent_unexpanded, axis=1)
print(f"\nkv_latent (after expand_dims axis=1): shape={kv_latent.shape}")

# Now apply RoPE
offset = 0
q_pe_rope = attn.rope(q_pe, offset)
k_pe_rope = attn.rope(k_pe, offset)
print(f"\nAfter RoPE:")
print(f"  q_pe: std_before={np.array(q_pe).astype(np.float64).std():.6f}  std_after={np.array(q_pe_rope).astype(np.float64).std():.6f}")
print(f"  k_pe: std_before={np.array(k_pe).astype(np.float64).std():.6f}  std_after={np.array(k_pe_rope).astype(np.float64).std():.6f}")

# Now the L>1 path: materialize k and v
print(f"\n--- L > 1 path (materialize k, v) ---")
print(f"  embed_q weight shape: {attn.embed_q.weight.shape}  (QuantizedMultiLinear)")
print(f"  unembed_out weight shape: {attn.unembed_out.weight.shape}")

k = attn.embed_q(kv_latent, transpose=False)
v = attn.unembed_out(kv_latent)
print(f"\n  k = embed_q(kv_latent, transpose=False): shape={k.shape}")
print(f"    std={np.array(k).astype(np.float64).std():.6f}  max(abs)={np.abs(np.array(k).astype(np.float64)).max():.6f}")
print(f"  v = unembed_out(kv_latent): shape={v.shape}")
print(f"    std={np.array(v).astype(np.float64).std():.6f}  max(abs)={np.abs(np.array(v).astype(np.float64)).max():.6f}")

# Compute pe_scores
pe_scores = (q_pe_rope * attn.scale) @ k_pe_rope.swapaxes(-1, -2)
print(f"\n  pe_scores = (q_pe_rope * scale) @ k_pe_rope.swapaxes(-1,-2): shape={pe_scores.shape}")
print(f"    scale={attn.scale}")
print(f"    std={np.array(pe_scores).astype(np.float64).std():.6f}  max={np.array(pe_scores).astype(np.float64).max():.4f}  min={np.array(pe_scores).astype(np.float64).min():.4f}")
print(f"    (Note: pe_scores dim = scale * q_pe@k_pe.T = scale * 64-dim dot)")

# Compute q_nope @ k.T (the actual content attention scores, before scale)
qk_nope = q_nope @ k.swapaxes(-1, -2)
print(f"\n  q_nope @ k.T (raw, before scale): shape={qk_nope.shape}")
print(f"    std={np.array(qk_nope).astype(np.float64).std():.6f}  max={np.array(qk_nope).astype(np.float64).max():.4f}  min={np.array(qk_nope).astype(np.float64).min():.4f}")
print(f"  After scaling by {attn.scale}: std={np.array(qk_nope).astype(np.float64).std()*attn.scale:.4f}")

# Add them (the proper total scores:)
total_scores = attn.scale * qk_nope + pe_scores
print(f"\n  Total attention scores (scale*q_nope@k.T + pe_scores): shape={total_scores.shape}")
print(f"    std={np.array(total_scores).astype(np.float64).std():.6f}  max={np.array(total_scores).astype(np.float64).max():.4f}  min={np.array(total_scores).astype(np.float64).min():.4f}")

# Verify sdpa can run with cache=None
print(f"\n  Calling scaled_dot_product_attention(q_nope, k, v, cache=None, scale, mask=pe_scores)...")
from mlx_lm.models.base import scaled_dot_product_attention
try:
    out = scaled_dot_product_attention(
        q_nope, k, v, cache=None, scale=attn.scale, mask=pe_scores
    )
    mx.eval(out)
    print(f"    SUCCESS — output shape={out.shape}")
    print(f"    std={np.array(out).astype(np.float64).std():.6f}  max(abs)={np.abs(np.array(out).astype(np.float64)).max():.6f}")
except Exception as e:
    print(f"    ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
