# Copyright © 2025 Apple Inc.
# GLM-5.2-specific override of the DeepseekV32 forward path.
#
# This subclass fixes the single confirmed forward-path divergence between
# GLM-5.2 and DeepSeek-V3.2: the MLA-internal layernorms (q_a_layernorm,
# kv_a_layernorm) are hardcoded to eps=1e-6 in deepseek_v32.Attention.__init__,
# but GLM-5.2 requires eps=config.rms_norm_eps (≈1e-5). Every other RMSNorm in
# the parent class already reads from config; only these two were missed.
#
# See docs/research/GLM52_MLX_MOE_DSA_DESIGN.md for the full gap analysis.
# RoPE (traditional=True = interleaved, matching rope_interleave=True), MoE gate
# (sigmoid + noaux_tc), attention math, and sanitize() are all inherited
# unchanged — they are already correct for GLM-5.2.

from dataclasses import dataclass
from typing import Dict, Optional

import mlx.nn as nn

from .base import BaseModelArgs
from .deepseek_v32 import Model as DSV32Model


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    topk_method: str
    scoring_func: str
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    num_experts_per_tok: int
    moe_layer_freq: int
    first_k_dense_replace: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_parameters: Dict
    attention_bias: bool
    # Traceability fields (optional; asserted for safety if present).
    rope_interleave: Optional[bool] = None
    expert_gating_func: Optional[int] = None
    rope_scaling: Dict = None
    rope_theta: Optional[float] = None

    def __post_init__(self):
        self.rope_scaling = self.rope_parameters
        self.rope_theta = self.rope_parameters["rope_theta"]
        # Safety assertions: if these GLM-5.2-specific fields are present,
        # they must match what the inherited forward path assumes.
        if self.rope_interleave is not None:
            assert self.rope_interleave is True, (
                "glm_moe_dsa forward uses traditional=True (interleaved RoPE); "
                f"config rope_interleave={self.rope_interleave} is incompatible."
            )
        if self.expert_gating_func is not None:
            # llama.cpp enum: 2 = SIGMOID. deepseek_v32 hardcodes mx.sigmoid.
            assert self.expert_gating_func == 2, (
                "glm_moe_dsa forward hardcodes sigmoid gating (func=2); "
                f"config expert_gating_func={self.expert_gating_func} unsupported."
            )


class Model(DSV32Model):
    """GLM-5.2 model with corrected MLA-internal RMSNorm eps.

    Overrides ``__init__`` to re-create the two MLA-internal layernorms
    (``q_a_layernorm``, ``kv_a_layernorm``) in every attention layer with
    ``eps=config.rms_norm_eps`` instead of the parent's hardcoded ``1e-6``.

    No ``__call__`` override — the entire DeepseekV32 forward graph is reused.
    Weights are loaded *after* ``__init__`` via ``load_weights`` / ``sanitize``
    (binding by dotted path), so re-creating these submodules here is safe: the
    trained ``weight`` vectors overwrite the fresh ones at load time.
    """

    def __init__(self, config: ModelArgs):
        super().__init__(config)
        eps = config.rms_norm_eps
        q_lora = config.q_lora_rank
        kv_lora = config.kv_lora_rank
        for layer in self.model.layers:
            sa = getattr(layer, "self_attn", None)
            if sa is None:
                continue  # skip MTP / non-attention layers
            if hasattr(sa, "q_a_layernorm"):
                sa.q_a_layernorm = nn.RMSNorm(q_lora, eps=eps)
            if hasattr(sa, "kv_a_layernorm"):
                sa.kv_a_layernorm = nn.RMSNorm(kv_lora, eps=eps)
