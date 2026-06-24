"""
GLM-5.2 (glm_moe_dsa) → JANGTQ_K Conversion
Adapted by the GLM-5.2 kitchen from jang_tools.convert_minimax_jangtq +
jang_tools.convert_glm51_jangtq_2l (both © Jinho Jang, eric@jangq.ai).

WHY THIS SCRIPT EXISTS
----------------------
There is no off-the-shelf GLM JANGTQ_K converter:
  * `jang convert` / `vmlx-engine convert` only expose fixed JANG_* tiers and
    affine JANG_*K k-quant — NOT the TurboQuant (MXTQ codebook) path.
  * The only GLM JANGTQ script (convert_glm51_jangtq_2l.py) is hard-wired to
    JANGTQ_2L (fixed 2-bit experts) and FP8 sources.
The user wants TurboQuant WITHOUT a fixed number of bits → JANGTQ_K, the only
non-uniform JANGTQ profile (per-projection mixed bits on routed experts).

DESIGN (validated against the vmlx-engine JANGTQ fast path)
-----------------------------------------------------------
Source: BF16 HuggingFace GLM-5.2 (`model_type=glm_moe_dsa`), per-expert tensors
  model.layers.L.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight

Bit policy (JANGTQ_K mixed):
  * routed experts  → MXTQ TurboQuant, per-projection:
        gate_proj  2-bit   (gated activation — less sensitive)
        up_proj    2-bit   (gated activation — less sensitive)
        down_proj  4-bit   (output enters the residual stream — most sensitive)
  * self_attn (MLA / DSA), shared_experts, embed_tokens, lm_head, MoE router
    `gate`, all norms / biases → FP16 passthrough.

WHY ATTENTION STAYS FP16 (this is the crux that fixes the earlier JANG_4L crash)
--------------------------------------------------------------------------------
glm_moe_dsa.Model subclasses deepseek_v32.Model. deepseek_v32.sanitize() does:
    quantized = (f"{prefix}.kv_b_proj.scales" in weights)
    ...
    if quantized:
        bits = (kv_b_proj.shape[-1] * 32) // kv_lora_rank   # ← can compute 1
        mx.quantize(..., bits=bits)                         # ← crashes for bits<2
By keeping attention FP16 (NO `.scales` written for kv_b_proj), `quantized` is
False, the whole quantize branch is skipped, and only the FP16 reshape into
embed_q / unembed_out runs. No 1-bit quantize → no crash.

LOADING
-------
The output `.tq_packed` keys route the bundle through vmlx-engine's native
JANGTQ fast path: vmlx_engine.utils.jang_loader._load_jang_v2 → MXTQ detection
→ jang_tools.load_jangtq.load_jangtq_model, which:
  * Explicitly supports GLM-5.1/5.2 `glm_moe_dsa` (see load_jangtq.py header).
  * Stacks per-expert `mlp.experts.E.{proj}` into 3D `switch_mlp.{proj}`
    TurboQuantSwitchLinear (glm_pat at load_jangtq.py:884).
  * Runs `model.sanitize()` only on the FP16 regular weights (attention etc.),
    which — per the note above — does NOT quantize kv_b_proj.

USAGE
-----
    python3 convert_glm52_jangtq_k.py <src_bf16_dir> <out_dir> [profile] [--clean]
        <src_bf16_dir>  BF16 HF GLM-5.2 directory (config.json + model-*.safetensors)
        <out_dir>       output directory for the JANGTQ bundle
        [profile]       JANGTQ_K (default, mixed), or JANGTQ2 / JANGTQ3 / JANGTQ4

The vMLX bundled Python MUST be used so jang_tools / mlx are importable:
    /Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3
"""
import sys
import json
import gc
import shutil
from pathlib import Path

import numpy as np
import mlx.core as mx
from tqdm import tqdm
from safetensors import safe_open
from safetensors.numpy import save_file

# NOTE: _load_bf16_tensor is imported from jang_tools.calibrate with a
# leading underscore (private symbol by convention). Vendored here as-is from
# the jang_tools version current at the time of the GLM-5.2 export work; if
# jang_tools' private API changes, this import is the breaking point.
# (REMEDIATION_PLAN P3 #1 — provenance comment; no behavior change.)
from jang_tools.calibrate import _load_bf16_tensor
from jang_tools.turboquant.linear import tq_quantize_weight

SEED = 42
MAX_SHARD = 1_000_000_000  # 1 GB per shard


# ── Profile / bit policy ───────────────────────────────────────────
def resolve_profile(profile_raw):
    """Return (canonical_profile, mixed_proj_bits_or_None, uniform_bits_or_None)."""
    p = profile_raw.upper()
    _PROFILE_BITS = {
        "JANGTQ2": 2, "JANGTQ_2L": 2, "JANGTQ_2S": 2,
        "JANGTQ3": 3, "JANGTQ_3L": 3, "JANGTQ_3S": 3,
        "JANGTQ4": 4, "JANGTQ_4M": 4, "JANGTQ_4K": 4,
        "JANGTQ_K": "mixed", "JANGTQK": "mixed",
    }
    if p not in _PROFILE_BITS:
        raise SystemExit(
            f"unknown profile {profile_raw!r}; expected one of {sorted(_PROFILE_BITS)}"
        )
    bits = _PROFILE_BITS[p]
    if bits == "mixed":
        # down_proj 4-bit (residual-stream output), gate/up 2-bit (gated).
        return "JANGTQ_K", {"gate_proj": 2, "up_proj": 2, "down_proj": 4}, None
    return f"JANGTQ{bits}", None, bits


def make_bits_and_method(mixed_proj_bits, uniform_bits):
    """Build the per-tensor (bits, method) classifier for GLM-5.2."""
    def get_bits_and_method(tensor_name):
        name = tensor_name.lower()

        # 1D norms / biases / router correction bias → FP16 passthrough
        if "norm" in name or tensor_name.endswith(".bias") \
                or "e_score_correction_bias" in name:
            return (16, "passthrough")

        # MoE router/gate (`.gate.weight`, NOT attention's gate_proj) → FP16
        if ".gate." in name and "gate_proj" not in name:
            return (16, "passthrough")

        # Embeddings / lm_head → FP16 (keeps logits + token table lossless)
        if "embed_tokens" in name or "lm_head" in name:
            return (16, "passthrough")

        # Attention (MLA / DSA) → FP16. CRITICAL: keeps kv_b_proj unquantized so
        # deepseek_v32.sanitize never hits the bits=1 mx.quantize crash.
        if "self_attn" in name:
            return (16, "passthrough")

        # Shared expert(s) → FP16 (always active, small, precision-worth it)
        if "shared_expert" in name:
            return (16, "passthrough")

        # Routed experts → MXTQ TurboQuant at the per-projection bit width.
        if "experts" in name and (
            "gate_proj" in name or "up_proj" in name or "down_proj" in name
        ):
            if mixed_proj_bits is not None:
                for proj_key, proj_bits in mixed_proj_bits.items():
                    if f".{proj_key}." in name or name.endswith(f".{proj_key}.weight"):
                        return (proj_bits, "mxtq")
                # Unrecognized expert-projection tensor name: do NOT silently
                # fall back to 2-bit MXTQ (would corrupt a future tensor variant
                # with zero signal). Surface the full name and abort so any new
                # projection type is handled explicitly in the mapping above.
                raise ValueError(
                    f"Unrecognized expert-projection tensor name: {tensor_name!r}. "
                    f"Expected one of {list(mixed_proj_bits)}; refusing to silently "
                    f"default to 2-bit MXTQ. Extend get_bits_and_method if this is a "
                    f"legitimate new projection type."
                )
            return (uniform_bits, "mxtq")

        # Anything unmapped → FP16 passthrough (safe default for GLM-5.2)
        return (16, "passthrough")

    return get_bits_and_method


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)

    SRC = Path(sys.argv[1])
    OUT = Path(sys.argv[2])
    # Positional profile is argv[3] unless that slot is the --clean flag.
    _positional = [a for a in sys.argv[3:] if not a.startswith("-")]
    PROFILE_RAW = _positional[0] if _positional else "JANGTQ_K"
    # When resuming, remove shard files with non-conforming names left over
    # from prior interrupted/aborted runs. Default OFF for safety (never delete
    # data without an explicit opt-in); CI should pass --clean for determinism.
    clean_orphans = "--clean" in sys.argv

    PROFILE, MIXED, UNIFORM = resolve_profile(PROFILE_RAW)
    get_bits_and_method = make_bits_and_method(MIXED, UNIFORM)

    OUT.mkdir(parents=True, exist_ok=True)

    with open(SRC / "config.json") as f:
        config = json.load(f)
    tc = config.get("text_config", config)
    n_layers = tc["num_hidden_layers"]
    first_dense = tc.get("first_k_dense_replace", 0)
    n_experts = tc.get("n_routed_experts", 256)

    print("=" * 64)
    print(f"  GLM-5.2 → {PROFILE} JANGTQ Conversion")
    print("=" * 64)
    print(f"  Source: {SRC}")
    print(f"  Output: {OUT}")
    print(f"  Layers: {n_layers} ({first_dense} dense, {n_layers - first_dense} MoE)")
    print(f"  Experts: {n_experts}")
    if MIXED is not None:
        print(f"  Profile: experts=MXTQ mixed "
              f"(gate={MIXED['gate_proj']}b, up={MIXED['up_proj']}b, "
              f"down={MIXED['down_proj']}b); attn/shared/embed/head/gate=fp16")
    else:
        print(f"  Profile: experts=MXTQ-{UNIFORM}b; attn/shared/embed/head/gate=fp16")
    print(flush=True)

    # ── Scan ──────────────────────────────────────────────────────
    print("  Scanning source...", flush=True)
    all_tensors = []
    for sf in sorted(SRC.glob("model-*.safetensors")):
        with safe_open(str(sf), framework="numpy") as f:
            for k in f.keys():
                if k.endswith("_scale_inv"):
                    continue
                shape = list(f.get_slice(k).get_shape())
                all_tensors.append((k, shape, sf))
    print(f"  Found {len(all_tensors)} tensors", flush=True)

    # ── Shard writer ──────────────────────────────────────────────
    state = {"shard_idx": 0, "shard_tensors": {}, "shard_bytes": 0}
    shard_map = {}

    def flush_shard():
        if not state["shard_tensors"]:
            return
        state["shard_idx"] += 1
        fname = f"model-{state['shard_idx']:05d}-of-XXXXX.safetensors"
        save_file(state["shard_tensors"], str(OUT / fname))
        for k in state["shard_tensors"]:
            shard_map[k] = fname
        print(f"    Shard {state['shard_idx']}: {len(state['shard_tensors'])} "
              f"tensors, {state['shard_bytes']/1e9:.2f} GB", flush=True)
        state["shard_tensors"] = {}
        state["shard_bytes"] = 0

    def add_tensor(name, arr):
        state["shard_tensors"][name] = arr
        state["shard_bytes"] += arr.nbytes
        if state["shard_bytes"] >= MAX_SHARD:
            flush_shard()

    # ── Resume support: collect keys already written ──────────────
    # Match BOTH the in-progress XXXXX placeholder name AND the finalized
    # NNNNN name so a resumed run detects its own completed (renamed) shards.
    # Without the NNNNN glob, a second invocation finds zero XXXXX shards,
    # sees done_keys={}, and re-quantizes every tensor — orphaning the prior
    # run's renamed shards (3 runs = 3x disk usage). See REMEDIATION_PLAN P1.
    import re as _re
    shard_name_re = _re.compile(r"^model-(\d+)-of-(\d+)\.safetensors$")
    existing = sorted(OUT.glob("model-*-of-*.safetensors"))
    done_keys = set()
    orphans = []  # shards not matching this run's emitted naming
    if existing:
        import struct as _struct
        print(f"  Resume: scanning {len(existing)} existing shards...", flush=True)
        for sf in existing:
            m = shard_name_re.match(sf.name)
            # Accept both placeholder (model-NNNNN-of-XXXXX) and finalized
            # (model-NNNNN-of-MMMMM) names; the numeric prefix is what shard_idx tracks.
            prefix_match = _re.match(r"^model-(\d+)-of-", sf.name)
            if not prefix_match:
                orphans.append(sf)
                continue
            with open(sf, "rb") as fh:
                hsize = _struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(hsize))
            for k in hdr:
                if k == "__metadata__":
                    continue
                done_keys.add(k)
                shard_map[k] = sf.name
            idx_str = prefix_match.group(1)
            state["shard_idx"] = max(state["shard_idx"], int(idx_str))
        print(f"  Resume: {len(done_keys)} keys present across "
              f"{len(existing) - len(orphans)} usable shards, continuing from "
              f"shard {state['shard_idx'] + 1}", flush=True)
        if orphans:
            if clean_orphans:
                for sf in orphans:
                    sf.unlink()
                print(f"  Resume: removed {len(orphans)} orphan shard(s) "
                      f"(non-conforming names)", flush=True)
            else:
                print(f"  Resume: WARNING {len(orphans)} orphan shard(s) with "
                      f"non-conforming names left in place (re-run with --clean "
                      f"to remove)", flush=True)

    def is_done(source_name, method):
        if method == "passthrough":
            return source_name in done_keys
        base = source_name[:-len(".weight")] if source_name.endswith(".weight") \
            else source_name
        return (f"{base}.tq_packed" in done_keys
                and f"{base}.tq_norms" in done_keys
                and f"{base}.tq_bits" in done_keys)

    # ── Convert ───────────────────────────────────────────────────
    print("\n  Converting...", flush=True)
    totals = {"mxtq": 0, "passthrough": 0, "skipped_resume": 0}
    processed = 0
    for tensor_name, shape, sf_path in tqdm(all_tensors, desc="  Processing"):
        bits, method = get_bits_and_method(tensor_name)

        if done_keys and is_done(tensor_name, method):
            totals["skipped_resume"] += 1
            totals[method if method == "passthrough" else "mxtq"] += 1
            continue

        # GLM-5.2 source is BF16; read raw via the bf16 loader. Fall back to
        # safe_open for any non-bf16 dtype (e.g. fp32 norms).
        try:
            tensor = _load_bf16_tensor(sf_path, tensor_name, shape)
        except Exception:
            with safe_open(str(sf_path), framework="numpy") as f:
                tensor = f.get_tensor(tensor_name)
                if not isinstance(tensor, np.ndarray):
                    tensor = np.array(tensor)
        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)

        if method == "passthrough":
            add_tensor(tensor_name, tensor.astype(np.float16))
            totals["passthrough"] += 1
        else:  # mxtq
            result = tq_quantize_weight(tensor, bits=bits, seed=SEED)
            base = tensor_name[:-len(".weight")] if tensor_name.endswith(".weight") \
                else tensor_name
            add_tensor(f"{base}.tq_packed", result["packed"])
            add_tensor(f"{base}.tq_norms", result["norms"])
            add_tensor(f"{base}.tq_bits", np.array([bits], dtype=np.uint8))
            totals["mxtq"] += 1

        del tensor
        processed += 1
        if processed % 200 == 0:
            gc.collect()

    flush_shard()

    if totals["skipped_resume"]:
        print(f"\n  Resume: skipped {totals['skipped_resume']} already-done tensors",
              flush=True)

    # ── Rename shards XXXXX → real total, write index ─────────────
    n_shards = state["shard_idx"]
    print(f"\n  Renaming {n_shards} shards + writing index...", flush=True)
    for i in range(1, n_shards + 1):
        old = OUT / f"model-{i:05d}-of-XXXXX.safetensors"
        new = OUT / f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        if old.exists():
            old.rename(new)
    shard_map_final = {k: v.replace("XXXXX", f"{n_shards:05d}")
                       for k, v in shard_map.items()}
    index = {"metadata": {"format": "jangtq", "total_size": 0},
             "weight_map": shard_map_final}
    with open(OUT / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # ── config.json + jang_config.json ────────────────────────────
    if MIXED is not None:
        routed_map = {
            "gate_proj": MIXED["gate_proj"],
            "up_proj":   MIXED["up_proj"],
            "down_proj": MIXED["down_proj"],
        }
        bits_default = MIXED["down_proj"]
    else:
        routed_map = UNIFORM
        bits_default = UNIFORM

    mxtq_bits_top = {
        "routed_expert": routed_map,
        "attention": 16,
        "shared_expert": 16,
        "embed_tokens": 16,
        "lm_head": 16,
        "norms_router_biases": 16,
    }

    config.pop("quantization_config", None)
    # bits=8 is only the affine-fallback default for the control plane; actual
    # routed bits live in mxtq_bits. (Matches every JANGTQ converter.)
    config["quantization"] = {
        "bits": 8,
        "mode": "affine",
        "group_size": 64,
        "routed_expert_bits": routed_map,
        "mxtq_bits": mxtq_bits_top,
    }
    config["mxtq_bits"] = mxtq_bits_top
    config["weight_format"] = "mxtq"
    with open(OUT / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    jang_config = {
        "version": 2,
        "format": "jang",
        "format_version": "2.0",
        "weight_format": "mxtq",
        "profile": PROFILE,
        "source_model": {
            "name": "GLM-5.2",
            "org": "zai-org",
            "architecture": "glm_moe_dsa",
        },
        "mxtq_seed": SEED,
        "mxtq_bits": mxtq_bits_top,
        "quantization": {
            "method": "passthrough+mxtq",
            "group_size": 64,
            "bits_default": bits_default,
        },
    }

    # Capabilities (glm_moe_dsa → glm5 family, MLA cache) for vmlx detector.
    try:
        from jang_tools.capabilities import build_capabilities, verify_directory
        caps = build_capabilities(jang_config, config, OUT)
        if caps is not None:
            jang_config["capabilities"] = caps
            print(f"  capabilities: family={caps['family']} "
                  f"reasoning={caps['reasoning_parser']} "
                  f"tool={caps['tool_parser']} cache={caps['cache_type']} "
                  f"modality={caps['modality']}", flush=True)
        else:
            print("  WARNING: capabilities unresolved; vmlx falls back to "
                  "silver/bronze.", flush=True)
    except Exception as _e:
        print(f"  capabilities step skipped: {_e}", flush=True)

    config["jang"] = jang_config
    with open(OUT / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(OUT / "jang_config.json", "w") as f:
        json.dump(jang_config, f, indent=2)

    # ── Copy tokenizer / templates / custom modeling .py ──────────
    for fn in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "generation_config.json", "chat_template.jinja",
               "chat_template.json", "merges.txt", "vocab.json",
               "preprocessor_config.json", "configuration.json"]:
        src_f = SRC / fn
        if src_f.exists():
            shutil.copy2(str(src_f), str(OUT / fn))
    for f in SRC.glob("*.py"):
        shutil.copy2(str(f), str(OUT / f.name))

    # ── Verify jang_config schema (catch drift early) ─────────────
    try:
        from jang_tools.capabilities import verify_directory
        ok, msg = verify_directory(OUT)
        print(f"  verify: {msg}", flush=True)
    except Exception as _e:
        print(f"  verify skipped: {_e}", flush=True)

    # ── Build Swift runtime sidecar (optional; Python loader doesn't need it) ─
    print("\n  Building jangtq_runtime.safetensors sidecar...", flush=True)
    try:
        from jang_tools.build_jangtq_sidecar import main as _build_sidecar
        _saved = sys.argv
        sys.argv = ["build_jangtq_sidecar", str(OUT)]
        try:
            _build_sidecar()
        finally:
            sys.argv = _saved
    except (Exception, SystemExit) as _e:
        print(f"  [sidecar] FAILED: {_e} — run "
              f"`python3 -m jang_tools.build_jangtq_sidecar {OUT}` manually "
              f"before any Swift-runtime use.", flush=True)

    du = sum(f.stat().st_size for f in OUT.glob("*") if f.is_file())
    print(f"\n{'='*64}")
    print(f"  DONE — GLM-5.2-{PROFILE}")
    print(f"  Output: {OUT}")
    print(f"    mxtq tensors:        {totals['mxtq']}")
    print(f"    passthrough tensors: {totals['passthrough']}")
    print(f"    shards:              {n_shards}")
    print(f"    total size:          {du/1e9:.1f} GB")
    print(f"{'='*64}", flush=True)


if __name__ == "__main__":
    main()
