"""Diagnostic v2: trace where the activation explosion happens in GLM-5.2 MLX.

For each decoder layer, print stats of (attention output, FFN output, residual).
Also verifies that q_a_layernorm / kv_a_layernorm weights are properly loaded
into the re-created modules.
"""
import argparse
import sys
import numpy as np
import mlx.core as mx
from mlx_lm import load


def stat(name, arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    flat = arr.flatten() if arr.ndim > 1 else arr
    if flat.size == 0:
        print(f"  {name:40s} | empty")
        return
    nan = np.isnan(flat).any()
    inf = np.isinf(flat).any()
    # use float64 to avoid overflow
    f64 = flat.astype(np.float64)
    print(f"  {name:40s} | shape={list(arr.shape)} mean={f64.mean():.4f} std={f64.std():.4f} "
          f"min={f64.min():.4f} max={f64.max():.4f} nan={nan} inf={inf}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-shortgpt-pruned-4bit-mlx")
    p.add_argument("--prompt", default="What is 2+2?")
    p.add_argument("--max-layer", type=int, default=8,
                   help="inspect layers 0..max_layer in detail (default 8)")
    args = p.parse_args()

    print(f"Loading: {args.model}", flush=True)
    model, tokenizer = load(args.model)
    print(f"Model: {type(model).__module__}.{type(model).__name__}")
    print(f"Layer type: {type(model.model.layers[0]).__name__}\n", flush=True)

    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        chat_template_kwargs={"enable_thinking": False, "reasoning_effort": None},
    )
    input_ids = mx.array(tokenizer.encode(prompt_text))[None]
    print(f"Tokens: {input_ids.shape[1]} | prompt: {repr(prompt_text[:80])}...\n")

    # Step 1: embeddings
    emb = model.model.embed_tokens(input_ids)
    mx.eval(emb)
    print("[1] EMBEDDINGS")
    stat("embed_tokens(out)", emb)

    # Verify q_a_layernorm weight is loaded (not the fresh "ones" init)
    print("[2] MLA NORM WEIGHT CHECK (layer 0)")
    l0 = model.model.layers[0]
    sa0 = l0.self_attn
    print(f"  q_a_layernorm type: {type(sa0.q_a_layernorm).__name__}")
    stat("  q_a_layernorm.weight", sa0.q_a_layernorm.weight)
    stat("  kv_a_layernorm.weight", sa0.kv_a_layernorm.weight)
    # Check: if all ones, weight wasn't loaded
    qa_w = np.array(sa0.q_a_layernorm.weight)
    kv_w = np.array(sa0.kv_a_layernorm.weight)
    if (qa_w == 1.0).all():
        print("  ⚠️  q_a_layernorm is all ones — weights NOT loaded into re-created module!")
    if (kv_w == 1.0).all():
        print("  ⚠️  kv_a_layernorm is all ones — weights NOT loaded into re-created module!")

    # Step 3: per-layer pass — call each sub-module separately
    print(f"\n[3] LAYER BY LAYER (0..{args.max_layer})")
    h = emb
    for i, layer in enumerate(model.model.layers):
        if i > args.max_layer:
            break
        # Save input to layer
        h_in = h
        # Compute attention + ffn separately
        residual = h
        h_normed = layer.input_layernorm(h)
        mx.eval(h_normed)
        # Run the attention block (NOT including its output projection composite)
        attn_out = layer.self_attn(h_normed, None)
        # In deepseek_v32, Attention.__call__ INCLUDES the o_proj and returns B,L,H.
        # In the DecoderLayer, the residual add is `h = residual + attn_out`
        h_post_attn = residual + attn_out
        # FFN block
        residual2 = h_post_attn
        h_normed2 = layer.post_attention_layernorm(h_post_attn)
        ffn_out = layer.mlp(h_normed2)
        h_post_ffn = residual2 + ffn_out
        mx.eval(h_post_ffn)

        print(f"  --- Layer {i} ({type(layer).__name__}) ---")
        stat("in",         h_in)
        stat("post_attn (resid+attn)", h_post_attn)
        stat("post_ffn (final)", h_post_ffn)
        h = h_post_ffn

    # Continue through remaining layers (just check every ~10)
    print(f"\n[4] SKIPPING THROUGH REST (sampled)")
    n_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        if i <= args.max_layer:
            continue
        h = layer(h, None)
        if i % 10 == 0 or i == n_layers - 1:
            mx.eval(h)
            stat(f"layer {i}", h)

    # Step: predict next token via lm_head directly
    print(f"\n[5] LM_HEAD")
    h_normed = model.model.norm(h)
    mx.eval(h_normed)
    stat("norm(out)", h_normed)
    logits = model.lm_head(h_normed)
    mx.eval(logits)
    stat("logits", logits)
    # Top-5 predicted next tokens
    last_logits = logits[0, -1, :]
    top = mx.argsort(-last_logits)[:10]
    print(f"\nTop-10 next tokens:")
    for tid in top:
        tok = tokenizer.decode([int(tid)])
        print(f"  {int(tid):6d} -> {repr(tok)}  (logit={float(last_logits[int(tid)]):.3f})")


if __name__ == "__main__":
    main()
