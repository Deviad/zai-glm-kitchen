"""Check: do embed_q (W_uk) and unembed_out (W_uv) weights have the right magnitude
to compensate for the tiny kv_latent produced by the small-gamma kv_a_layernorm?

Theory: in MLA, kv_latent = gamma * (compressed_kv / RMS(compressed_kv)).
If gamma is tiny (0.006), kv_latent will be tiny. Then embed_q(kv_latent) = W_uk @ kv_latent
will be tiny. So q_nope @ k.T will be tiny while pe_scores are normal magnitude.

For the model to actually work, W_uk must have large magnitudes (1000x) to compensate.
Let's check.
"""
import json
import numpy as np
from pathlib import Path
from gguf import GGUFReader, dequantize
from safetensors import safe_open
import mlx.core as mx

GGUF_DIR = Path("/Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-shortgpt-pruned-IQ2S-experts-IQ4NL-rest")
MLX_DIR = "/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-shortgpt-pruned-4bit-mlx"

shard2 = sorted(GGUF_DIR.glob("*.gguf"))[1]
r = GGUFReader(str(shard2))
ts = {t.name: t for t in r.tensors}

idx = json.load(open(f"{MLX_DIR}/model.safetensors.index.json"))
wm = idx["weight_map"]


def mlx_load(key):
    shard = wm[key]
    with safe_open(f"{MLX_DIR}/{shard}", framework="numpy") as st:
        return st.get_tensor(key)


def mlx_dequant(key, bits=4, group_size=64):
    """Load a quantized MLX tensor and dequantize it."""
    w = mlx_load(f"{key}.weight")
    s = mlx_load(f"{key}.scales")
    b = mlx_load(f"{key}.biases")
    return np.array(mx.dequantize(mx.array(w), mx.array(s), mx.array(b),
                                  bits=bits, group_size=group_size))


def gguf_dequant(name):
    t = ts[name]
    return np.array(dequantize(t.data, t.tensor_type), copy=False)


print("=" * 70)
print("=== LAYER 0 EMBED_Q / UNEMBED_OUT (W_uk / W_uv) MAGNITUDES ===")
print("=" * 70)
print()
print("kv_a_layernorm.weight (GGUF):", end=" ")
g_kv_norm = gguf_dequant("blk.0.attn_kv_a_norm.weight")
print(f"shape={g_kv_norm.shape} mean={g_kv_norm.mean():.6f} std={g_kv_norm.std():.6f} min={g_kv_norm.min():.6f} max={g_kv_norm.max():.6f}")
print("kv_a_layernorm.weight (MLX): ", end=" ")
m_kv_norm = mlx_load("model.layers.0.self_attn.kv_a_layernorm.weight")
m_kv_norm_f = np.array(m_kv_norm).astype(np.float32)
print(f"shape={m_kv_norm_f.shape} mean={m_kv_norm_f.mean():.6f} std={m_kv_norm_f.std():.6f} min={m_kv_norm_f.min():.6f} max={m_kv_norm_f.max():.6f}")
print()

# Check GGUF k_b / v_b shapes and values
print("--- GGUF blk.0.attn_k_b (3-D tensor: shape preserved after dequant) ---")
g_k = gguf_dequant("blk.0.attn_k_b.weight")
print(f"  shape={g_k.shape} mean={g_k.mean():.4f} std={g_k.std():.4f} min={g_k.min():.4f} max={g_k.max():.4f}")

print("--- GGUF blk.0.attn_v_b ---")
g_v = gguf_dequant("blk.0.attn_v_b.weight")
print(f"  shape={g_v.shape} mean={g_v.mean():.4f} std={g_v.std():.4f} min={g_v.min():.4f} max={g_v.max():.4f}")

# MLX embed_q and unembed_out (weight shape (num_heads=64, output_dims, input_dims))
# Need to dequantize them
print("\n--- MLX embed_q (W_uk): quantized MultiLinear, weight shape (64, 512, 24 packed = unpack to 192) ---")
try:
    m_eq = mlx_dequant("model.layers.0.self_attn.embed_q", bits=4, group_size=64)
    print(f"  dequant shape={m_eq.shape}")
    print(f"  stats: mean={m_eq.mean():.4f} std={m_eq.std():.4f} min={m_eq.min():.4f} max={m_eq.max():.4f}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

print("\n--- MLX unembed_out (W_uv): quantized MultiLinear ---")
try:
    m_uo = mlx_dequant("model.layers.0.self_attn.unembed_out", bits=4, group_size=64)
    print(f"  dequant shape={m_uo.shape}")
    print(f"  stats: mean={m_uo.mean():.4f} std={m_uo.std():.4f} min={m_uo.min():.4f} max={m_uo.max():.4f}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# GGUF k_b has shape (192, 512, 64) = (dk_nope, kv_lora, n_head)
# MLX embed_q has shape (64, 512, 192) = (n_head, output_dims=kv_lora=512, input_dims=dk_nope=192)
# Wait — let me check the constructor:
#   MultiLinear(input_dims=qk_nope_head_dim=192, output_dims=kv_lora_rank=512, num_heads=64)
# So weight shape (num_heads=64, output_dims=512, input_dims=192).
# When called with transpose=False: x @ weight → x[*, 192] @ weight[*, 512, 192] → result [*, 64, 512]?
# Hmm actually:
#   weight shape (64, 512, 192)
#   x shape (1, 1, 21, 512) (kv_latent)
#   we want k = x @ weight: x[*, 512] @ weight[64, 512, 192]?
#   MLX matmul broadcasts over leading dims. Last-2 dims of x (21, 512), last-2 of weight (512, 192). Contract 512 with 512 → OK
#   So result shape: broadcast of leading (1, 1) with (64,) → (1, 1, 64, 21, 192)? wait that's what I saw earlier
# Actually MLX test showed: (1, 64, 21, 192) — let me re-check.

print("\n=== Verify the matmul semantics with actual MLX calls ===")
from mlx_lm import load
import time
print(f"Loading model... ({time.strftime('%H:%M:%S')})", flush=True)
t0 = time.time()
model, _ = load(MLX_DIR)
print(f"  loaded in {time.time()-t0:.1f}s")

sa = model.model.layers[0].self_attn
import inspect
print(f"\nembed_q class: {type(sa.embed_q).__name__}")
eq_w = sa.embed_q.weight
print(f"  raw weight shape (packed): {eq_w.shape}")
print(f"  scales shape: {sa.embed_q.scales.shape}")

# Compute k = embed_q(kv_latent, transpose=False) for some sample kv_latent
# Use a random kv_latent matching the model's actual output
kv_latent = mx.array(np.random.randn(1, 1, 21, 512).astype(np.float32) * 0.005)
print(f"\nSample kv_latent: shape={kv_latent.shape} std={np.array(kv_latent).std():.4f}")
k = sa.embed_q(kv_latent, transpose=False)
print(f"k = embed_q(kv_latent, transpose=False): shape={k.shape}")
mx.eval(k)
print(f"  std={np.array(k).astype(np.float64).std():.6f}  max={np.abs(np.array(k).astype(np.float64)).max():.6f}")

# Now compare: this gives k as (1, 1, 64, 21, 192) or (1, 64, 21, 192)?
print(f"\nk shape: {k.shape}")
