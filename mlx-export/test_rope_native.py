"""Verify RoPE norm-preservation on real model activations."""
import numpy as np
import mlx.core as mx
from mlx_lm import load

OUT = "/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-shortgpt-pruned-4bit-mlx"
model, tokenizer = load(OUT)
messages = [{"role": "user", "content": "What is 2+2?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
    chat_template_kwargs={"enable_thinking": False, "reasoning_effort": None})
ids = mx.array(tokenizer.encode(prompt))[None]
h = model.model.embed_tokens(ids)

# Layer 0 attention
attn = model.model.layers[0].self_attn
norm_h = model.model.layers[0].input_layernorm(h)
B, L, D = norm_h.shape
qr = attn.q_a_layernorm(attn.q_a_proj(norm_h))
q = attn.q_b_proj(qr)
q = q.reshape(B, L, attn.num_heads, attn.q_head_dim).transpose(0, 2, 1, 3)
q_nope, q_pe = mx.split(q, [attn.qk_nope_head_dim], axis=-1)
mx.eval(q_pe)

# Track per-pair and overall norm BEFORE RoPE
qp_np = np.array(q_pe).astype(np.float64)
print(f"q_pe shape: {qp_np.shape}")
print(f"  before RoPE: std={qp_np.std():.6f}  ||total||²={np.sum(qp_np**2):.4f}")
print(f"  pair 0 norm before: {np.linalg.norm(qp_np[..., 0:2]):.6f}")
print(f"  pair 1 norm before: {np.linalg.norm(qp_np[..., 2:4]):.6f}")
print(f"  pair 31 norm before: {np.linalg.norm(qp_np[..., 62:64]):.6f}")

q_pe_rot = attn.rope(q_pe, offset=0)
mx.eval(q_pe_rot)
qpr_np = np.array(q_pe_rot).astype(np.float64)
print(f"\n  after RoPE:  std={qpr_np.std():.6f}  ||total||²={np.sum(qpr_np**2):.4f}")
print(f"  pair 0 norm after:  {np.linalg.norm(qpr_np[..., 0:2]):.6f}")
print(f"  pair 1 norm after:  {np.linalg.norm(qpr_np[..., 2:4]):.6f}")
print(f"  pair 31 norm after: {np.linalg.norm(qpr_np[..., 62:64]):.6f}")

print(f"\n  ratio std: {qpr_np.std()/qp_np.std():.6f}")
print(f"  ratio ||total||²: {np.sum(qpr_np**2)/np.sum(qp_np**2):.6f}")

# Now k_pe: it has shape (B, 1, L, 64) after reshape, broadcast in pe_scores
ckv = attn.kv_a_proj_with_mqa(norm_h)
ckv, k_pe = mx.split(ckv, [attn.kv_lora_rank], axis=-1)
k_pe = k_pe.reshape(B, L, 1, attn.qk_rope_head_dim).transpose(0, 2, 1, 3)
mx.eval(k_pe)
kp_np = np.array(k_pe).astype(np.float64)
print(f"\nk_pe shape: {kp_np.shape}")
print(f"  before RoPE: std={kp_np.std():.6f}  ||total||²={np.sum(kp_np**2):.4f}")
k_pe_rot = attn.rope(k_pe, offset=0)
mx.eval(k_pe_rot)
kpr_np = np.array(k_pe_rot).astype(np.float64)
print(f"  after RoPE:  std={kpr_np.std():.6f}  ||total||²={np.sum(kpr_np**2):.4f}")
print(f"  ratio std: {kpr_np.std()/kp_np.std():.6f}")
