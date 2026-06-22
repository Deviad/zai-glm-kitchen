"""Test if MLX RoPE preserves norms for traditional=True (interleaved)."""
import mlx.core as mx
import numpy as np
from mlx_lm.models.rope_utils import initialize_rope

# Same params as GLM-5.2 config
rope = initialize_rope(
    dims=64,
    base=8_000_000.0,
    traditional=True,
    max_position_embeddings=32768,
    scaling_config=None,
)

print("=== Test 1: random input, std=1 ===")
for L in [1, 8, 21, 128]:
    x = mx.random.normal((1, 1, L, 64))  # std~1
    y = rope(x, offset=0)
    print(f"L={L:4d}  in std={mx.std(x):.5f}  out std={mx.std(y):.5f}  ratio={float(mx.std(y)/mx.std(x)):.4f}  in max={float(mx.abs(x).max()):.5f}  out max={float(mx.abs(y).max()):.5f}")

# Norm check per pair
print("\n=== Test 2: pair-norm preservation ===")
x = mx.array([[[[1.0, 2.0, 3.0, 4.0]]]])  # L=1, 2 pairs: (1,2), (3,4)
y = rope(x, offset=0)
xn = np.array(x).flatten(); yn = np.array(y).flatten()
print(f"in: {xn}")
print(f"out: {yn}")
print(f"  pair0 norm in={np.linalg.norm(xn[:2]):.6f}  out={np.linalg.norm(yn[:2]):.6f}")
print(f"  pair1 norm in={np.linalg.norm(xn[2:4]):.6f}  out={np.linalg.norm(yn[2:4]):.6f}")

# With offset > 0 (decode mode)
print("\n=== Test 3: offset > 0 (decode at position N) ===")
for offset in [0, 10, 100, 1000, 10000]:
    x = mx.random.normal((1, 1, 1, 64))
    y = rope(x, offset=offset)
    print(f"offset={offset:6d}  in std={mx.std(x):.5f}  out std={mx.std(y):.5f}  ratio={float(mx.std(y)/mx.std(x)):.4f}")
