"""Mixed-precision quantization predicate for GLM-5.2 streaming converter.

Mirrors the GGUF source policy (IQ2_S experts / IQ4_NL rest) in MLX affine:
  - Routed expert linears  → 2-bit affine (group_size=64)
  - Other linears (attn, shared experts, indexer) → 4-bit affine (group_size=64)
  - Norms, embeddings, router, e_score_correction_bias, lm_head → fp16 (skip)

The predicate operates on HF-format tensor names (the names that land in
safetensors BEFORE mlx-lm's sanitize() runs). This matches the per-path key
scheme that mlx-lm's quantize_model() writes into config.json["quantization"].
"""

GROUP_SIZE = 64
MODE = "affine"

# Name fragments that should NEVER be quantized (kept in fp16)
FP16_FRAGMENTS = (
    "norm",               # input_layernorm, post_attention_layernorm, q_a_layernorm, kv_a_layernorm, k_norm
    "embed_tokens",       # token embeddings
    "lm_head",            # output projection
    "e_score_correction_bias",  # noaux_tc router bias
    "gate.weight",        # MoE router (gate.weight, NOT gate_proj)
)

# Shared experts look like "mlp.shared_experts.gate_proj" — they should get 4-bit,
# NOT 2-bit. Routed experts are "mlp.experts.E.gate_proj" — those get 2-bit.
ROUTED_EXPERT_FRAGMENT = ".experts."


def quant_rule(hf_name: str):
    """Return quantization params dict, or False to keep fp16.

    Args:
        hf_name: HF-format tensor name (e.g. "model.layers.0.mlp.experts.3.gate_proj.weight")

    Returns:
        dict like {"bits": 2, "group_size": 64, "mode": "affine"} for quantized tensors
        False for fp16 tensors
    """
    # Never quantize norms, embeddings, router, biases
    for frag in FP16_FRAGMENTS:
        if frag in hf_name:
            return False

    # Only quantize .weight tensors (not .bias, .scales, etc.)
    if not hf_name.endswith(".weight"):
        return False

    # Routed expert linears → 2-bit
    if ROUTED_EXPERT_FRAGMENT in hf_name and "shared_experts" not in hf_name:
        return {"bits": 2, "group_size": GROUP_SIZE, "mode": MODE}

    # All other linears (attention, shared experts, indexer, kv_b_proj) → 4-bit
    # This catches: q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj,
    #               shared_experts.{gate,up,down}_proj,
    #               indexer.{wq_b, wk, weights_proj}
    return {"bits": 4, "group_size": GROUP_SIZE, "mode": MODE}
