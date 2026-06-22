"""Inspect attention at L0, L4, L7 (compare)."""
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention

OUT = "/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-shortgpt-pruned-4bit-mlx"
model, tokenizer = load(OUT)
messages = [{"role": "user", "content": "What is 2+2?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
    chat_template_kwargs={"enable_thinking": False, "reasoning_effort": None})
ids = mx.array(tokenizer.encode(prompt))[None]
h0 = model.model.embed_tokens(ids)
mx.eval(h0)

# Inspect attention at each target layer
for target in [0, 4, 7]:
    h = h0
    for i in range(target):
        h = model.model.layers[i](h, None, None)
    mx.eval(h)
    print(f"\n=== Layer {target} (input std={np.array(h).astype(np.float64).std():.5f}, max={np.abs(np.array(h).astype(np.float64)).max():.5f}) ===")

    attn = model.model.layers[target].self_attn
    norm_h = model.model.layers[target].input_layernorm(h)
    mx.eval(norm_h)
    print(f"  input_layernorm out: std={np.array(norm_h).astype(np.float64).std():.5f}  max={np.abs(np.array(norm_h).astype(np.float64)).max():.5f}")
    B, L, D = norm_h.shape
    qr = attn.q_a_layernorm(attn.q_a_proj(norm_h))
    mx.eval(qr)
    print(f"  q_a_layernorm(0):  std={np.array(qr).astype(np.float64).std():.5f}  max={np.abs(np.array(qr).astype(np.float64)).max():.5f}")
    q = attn.q_b_proj(qr)
    mx.eval(q)
    print(f"  q_b_proj out:       std={np.array(q).astype(np.float64).std():.5f}  max={np.abs(np.array(q).astype(np.float64)).max():.5f}")
    q = q.reshape(B, L, attn.num_heads, attn.q_head_dim).transpose(0, 2, 1, 3)
    q_nope, q_pe = mx.split(q, [attn.qk_nope_head_dim], axis=-1)
    ckv = attn.kv_a_proj_with_mqa(norm_h)
    mx.eval(ckv)
    print(f"  ckv out:            std={np.array(ckv).astype(np.float64).std():.5f}  max={np.abs(np.array(ckv).astype(np.float64)).max():.5f}")
    ckv, k_pe = mx.split(ckv, [attn.kv_lora_rank], axis=-1)
    kv_latent = attn.kv_a_layernorm(ckv)
    mx.eval(kv_latent)
    print(f"  kv_a_layernorm:    std={np.array(kv_latent).astype(np.float64).std():.5f}  max={np.abs(np.array(kv_latent).astype(np.float64)).max():.5f}")
    print(f"  k_pe:               std={np.array(k_pe).astype(np.float64).std():.5f}  max={np.abs(np.array(k_pe).astype(np.float64)).max():.5f}")
    q_pe_rot = attn.rope(q_pe, offset=0)
    k_pe_rot = attn.rope(k_pe, offset=0)
    mx.eval(q_pe_rot, k_pe_rot)
    print(f"  q_pe_rope:          std={np.array(q_pe_rot).astype(np.float64).std():.5f}  max={np.abs(np.array(q_pe_rot).astype(np.float64)).max():.5f}")
    pe_scores = (q_pe * attn.scale) @ k_pe.swapaxes(-1, -2)
    mx.eval(pe_scores)
    print(f"  pe_scores:          std={np.array(pe_scores).astype(np.float64).std():.5f}  max={np.array(pe_scores).astype(np.float64).max():.5f}  min={np.array(pe_scores).astype(np.float64).min():.5f}")
    mask = mx.tril(mx.ones((L, L), dtype=mx.bool_))
    pe_scores_masked = mx.where(mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype))
    mx.eval(pe_scores_masked)
    k = attn.embed_q(kv_latent[..., None, :, :], transpose=False)
    v = attn.unembed_out(kv_latent[..., None, :, :])
    mx.eval(k, v)
    print(f"  k (after embed_q):  std={np.array(k).astype(np.float64).std():.5f}  max={np.abs(np.array(k).astype(np.float64)).max():.5f}")
    print(f"  v (unembed_out):   std={np.array(v).astype(np.float64).std():.5f}  max={np.abs(np.array(v).astype(np.float64)).max():.5f}")
    out = scaled_dot_product_attention(q_nope, k, v, cache=None, scale=attn.scale, mask=pe_scores_masked)
    mx.eval(out)
    print(f"  attn output:        std={np.array(out).astype(np.float64).std():.5f}  max={np.abs(np.array(out).astype(np.float64)).max():.5f}")
    out_proj = attn.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))
    mx.eval(out_proj)
    print(f"  o_proj:             std={np.array(out_proj).astype(np.float64).std():.5f}  max={np.abs(np.array(out_proj).astype(np.float64)).max():.5f}")
