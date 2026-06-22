"""Streaming mixed-precision GGUF → MLX converter for GLM-5.2.

Reads multi-shard GGUF → per-tensor dequant → mx.quantize per predicate →
streaming sharded safetensors. No fp16 intermediate on disk.

Reuses gguf2mlx's name-mapping and MLA-transform helpers (expert split, kv_b
reconstruction) so the output matches what mlx-lm's deepseek_v32.sanitize()
expects on load.

Usage:
    python gguf_to_mlx_streaming.py \\
        --gguf-dir /path/to/GLM-5.2-shortgpt-pruned-... \\
        --output /path/to/GLM-5.2-shortgpt-pruned-mixed-mlx
"""

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import mlx.core as mx
from gguf import GGUFReader, GGMLQuantizationType, dequantize as gguf_dequant

# Reuse gguf2mlx helpers for name mapping + MLA transforms + config building
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "gguf2mlx" / "src"))
from gguf2mlx.gguf2mlx import (  # noqa: E402
    detect_architecture,
    build_config,
    extract_tokenizer,
    _read_mla_dims,
    _plan_tensor_emit,
    get_metadata_str,
    get_metadata_int,
)
from predicate import quant_rule  # noqa: E402

MAX_SHARD_BYTES = 5 * 1024**3  # 5 GB (mlx-lm default)


def _try_meta_int(reader, arch: str, key: str) -> Optional[int]:
    """Try {arch}.{key} then llama.{key}."""
    return (get_metadata_int(reader, f"{arch}.{key}")
            or get_metadata_int(reader, f"llama.{key}"))


# ─── MLA 3D k_b/v_b → 2D kv_b_proj combination ────────────────────────────────

def combine_kv_b_3d(k_b: np.ndarray, v_b: np.ndarray) -> np.ndarray:
    """Combine 3D MLA k_b/v_b into 2D kv_b_proj for mlx-lm's sanitize().

    gguf.dequantize returns MLA k_b/v_b in transposed 3D:
      k_b: [n_heads, kv_lora, dk_nope]
      v_b: [n_heads, dv, kv_lora]

    mlx-lm's sanitize() expects kv_b_proj.weight as 2D
    [n_heads*(dk_nope+dv), kv_lora], which it reshapes to
    [n_heads, dk_nope+dv, kv_lora] and splits into embed_q + unembed_out.
    """
    n_heads, kv_lora, dk_nope = k_b.shape
    dv = v_b.shape[1]
    assert v_b.shape[0] == n_heads and v_b.shape[2] == kv_lora, \
        f"v_b shape {v_b.shape} incompatible with k_b {k_b.shape}"
    k = np.transpose(k_b, (0, 2, 1))  # [n_heads, dk_nope, kv_lora]
    kv = np.concatenate([k, v_b], axis=1)  # [n_heads, dk_nope+dv, kv_lora]
    return kv.reshape(n_heads * (dk_nope + dv), kv_lora).astype(k_b.dtype)


# ─── per-tensor dequant (mirrors gguf2mlx's dequant switch) ───────────────────

def dequant_tensor(tensor) -> np.ndarray:
    """Dequantize a single GGUF tensor to fp16 numpy in HF layout [out, in].

    IMPORTANT: gguf's GGUFReader exposes ``tensor.data`` ALREADY shaped as the
    reversed logical shape — i.e. ``(out, in)`` for a 2D weight — which is
    exactly the HF/PyTorch ``(out_features, in_features)`` convention and the
    same layout ``gguf.dequantize`` returns for quantized blocks. We must NOT
    ``.reshape(logical_shape)`` (that reinterprets ``(out, in)`` memory as
    ``(in, out)`` and scrambles the values) and must NOT ``.T`` afterwards.

    The historical reshape+transpose silently corrupted every 2D F32 tensor —
    in GLM-5.2 that is the MoE router ``ffn_gate_inp.weight`` for every MoE
    layer (L3..L65), which destroyed expert routing → near-uniform expert
    selection → flat logits → gibberish generation. See GLM52_SESSION_MEMORY.md
    "Bug: F32 2D router reshape+transpose".
    """
    qtype_val = int(tensor.tensor_type)
    raw_data = tensor.data

    if qtype_val == 0:  # F32
        # raw_data is already (out, in) for 2D / (n,) for 1D — use as-is.
        return np.array(raw_data, dtype=np.float32).astype(np.float16)

    elif qtype_val == 1:  # F16
        return np.array(raw_data, dtype=np.float16)

    elif qtype_val == 28:  # F64
        return np.array(raw_data, dtype=np.float64).astype(np.float16)

    elif qtype_val in (24, 25, 26, 27):  # I8, I16, I32, I64
        int_map = {24: np.int8, 25: np.int16, 26: np.int32, 27: np.int64}
        arr = np.array(raw_data, dtype=int_map.get(qtype_val, np.int32))
        return arr.astype(np.float16)

    else:
        # Quantized block (IQ2_S=22, IQ4_NL=20, Q6_K=14, etc.)
        # gguf.dequantize already returns HF layout [out, in] for 2D
        ggml_qtype = GGMLQuantizationType(qtype_val)
        arr = gguf_dequant(raw_data, ggml_qtype)
        return arr.astype(np.float16)


# ─── streaming shard writer ───────────────────────────────────────────────────

class ShardWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.buffer: dict[str, mx.array] = {}
        self.current_bytes = 0
        self.shard_idx = 0
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0

    def add(self, name: str, arr: mx.array):
        self.buffer[name] = arr
        self.current_bytes += arr.nbytes
        if self.current_bytes >= MAX_SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        self.shard_idx += 1
        shard_name = f"model-{self.shard_idx:05d}-of-XXXXX.safetensors"
        shard_path = self.output_dir / shard_name
        mx.save_safetensors(str(shard_path), self.buffer, metadata={"format": "mlx"})
        for k in self.buffer:
            self.weight_map[k] = shard_name
        self.total_bytes += sum(v.nbytes for v in self.buffer.values())
        n = len(self.buffer)
        self.buffer.clear()
        self.current_bytes = 0  # ← reset accumulator after flush
        gc.collect()
        print(f"    ✓ Shard {self.shard_idx}: {n} tensors, "
              f"{self.total_bytes / 1e9:.2f} GB total so far")

    def finalize(self) -> int:
        """Flush remaining buffer; rename shards; write index. Return shard count."""
        self.flush()
        total_shards = self.shard_idx

        # Rename shards: replace XXXXX placeholder with actual count
        for i in range(1, total_shards + 1):
            old = self.output_dir / f"model-{i:05d}-of-XXXXX.safetensors"
            new = self.output_dir / f"model-{i:05d}-of-{total_shards:05d}.safetensors"
            old.rename(new)
            # Update weight_map
            for k, v in list(self.weight_map.items()):
                if v == old.name:
                    self.weight_map[k] = new.name

        # Write index
        index = {
            "metadata": {"total_size": self.total_bytes},
            "weight_map": dict(sorted(self.weight_map.items())),
        }
        with open(self.output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
        return total_shards


# ─── main converter ───────────────────────────────────────────────────────────

def convert_streaming(gguf_dir: str, output_dir: str, dry_run: int = 0,
                      reference_tokenizer: Optional[str] = None):
    """Convert multi-shard GGUF → mixed-precision MLX. No fp16 intermediate."""
    gguf_dir = Path(gguf_dir)
    output_dir = Path(output_dir)
    shards = sorted(gguf_dir.glob("*.gguf"))
    if not shards:
        print(f"❌ No .gguf files in {gguf_dir}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Read metadata from shard 1 ──
    print("=" * 60)
    print("Streaming GGUF → MLX Mixed-Precision Converter")
    print(f"  Source:  {gguf_dir} ({len(shards)} shards)")
    print(f"  Output:  {output_dir}")
    print("=" * 60)

    t0 = time.time()
    reader_meta = GGUFReader(str(shards[0]))
    arch = detect_architecture(reader_meta)
    config = build_config(reader_meta, arch)
    mla_dims = _read_mla_dims(reader_meta, arch)
    model_name = get_metadata_str(reader_meta, "general.name") or gguf_dir.name

    # Fix MLA v_head_dim: GGUF has both value_length (512, full) and
    # value_length_mla (256, actual MLA per-head v dim). The model's sanitize()
    # reshapes kv_b_proj using v_head_dim, so it MUST match the tensor data (256).
    v_head_mla = _try_meta_int(reader_meta, arch, "attention.value_length_mla")
    if v_head_mla:
        config["v_head_dim"] = v_head_mla

    # Fix n_routed_experts: GGUF uses expert_count → config num_experts,
    # but mlx-lm's ModelArgs expects n_routed_experts.
    if "num_experts" in config and "n_routed_experts" not in config:
        config["n_routed_experts"] = config["num_experts"]

    # Add rope_parameters: mlx-lm's glm_moe_dsa.ModelArgs requires a nested
    # dict (not flat rope_theta). __post_init__ extracts rope_theta from it.
    if "rope_parameters" not in config:
        config["rope_parameters"] = {
            "rope_theta": config.get("rope_theta", 10000.0),
            "rope_scaling": config.get("rope_scaling"),
            "rope_type": "default",
        }

    print(f"\n[1/4] Architecture: {arch}, model: {model_name}")
    print(f"  Config: {config.get('num_hidden_layers')} layers, "
          f"{config.get('hidden_size')} hidden, "
          f"{config.get('n_routed_experts', '?')} experts, "
          f"v_head_dim={config.get('v_head_dim')}")

    # ── Step 2: Write config.json (with quantization block, filled per-tensor) ──
    quantization_block = {
        "group_size": 64,
        "bits": 4,           # default
        "mode": "affine",
        # per-path overrides added during tensor loop
    }

    # ── Step 3: Extract tokenizer ──
    print("\n[2/4] Extracting tokenizer...")
    extract_tokenizer(reader_meta, output_dir, arch)
    # The GGUF-derived tokenizer.json is INCOMPLETE for GLM-5.2: it carries only
    # ~25 added_tokens and DROPS the special thinking/tool tokens (<think>=154841,
    # </think>=154842, <tool_call>, ...). The model then ENCODES the literal text
    # "<think></think>" as BPE pieces instead of the two atomic special tokens,
    # corrupting the generation-start position. If a known-good reference
    # tokenizer.json (original GLM-5.2 HF release / a prior good MLX export) is
    # provided, copy it over the extracted one.
    if reference_tokenizer:
        import shutil
        ref = Path(reference_tokenizer)
        ref_json = ref / "tokenizer.json" if ref.is_dir() else ref
        if ref_json.is_file():
            shutil.copy(str(ref_json), str(output_dir / "tokenizer.json"))
            print(f"    + overwrote tokenizer.json from reference: {ref_json}")
        else:
            print(f"    ! reference tokenizer not found: {ref_json} (kept GGUF one)")

    # ── Step 4: Stream tensors across all shards ──
    print(f"\n[3/4] Streaming + quantizing tensors{' (DRY RUN: first ' + str(dry_run) + ')' if dry_run else ''}...")
    writer = ShardWriter(output_dir)
    pending_kv_b: dict = {}
    total_in, total_out, tensor_count = 0, 0, 0
    quantized_count, fp16_count = 0, 0

    for shard_idx, shard_path in enumerate(shards, 1):
        if dry_run and tensor_count >= dry_run:
            break
        reader = GGUFReader(str(shard_path))
        if len(reader.tensors) == 0:
            print(f"  [shard {shard_idx}/{len(shards)}] {shard_path.name}: metadata-only, skipping")
            continue
        print(f"  [shard {shard_idx}/{len(shards)}] {shard_path.name}: {len(reader.tensors)} tensors")

        for tensor in reader.tensors:
            if dry_run and tensor_count >= dry_run:
                break
            tensor_count += 1
            total_in += tensor.n_bytes

            # Dequantize
            arr = dequant_tensor(tensor)
            name = tensor.name

            # Intercept MLA k_b/v_b: gguf.dequantize returns them as 3D,
            # but _plan_tensor_emit's _reconstruct_kv_b expects 2D with
            # correct MLA dims. Handle the combination directly.
            m_kv = re.match(r"blk\.(\d+)\.attn_(k_b|v_b)(?:\.weight)?$", name)
            if m_kv and arch in ("glm-dsa", "deepseek2", "deepseek3", "glm4moe"):
                layer_idx = m_kv.group(1)
                which = "k" if m_kv.group(2) == "k_b" else "v"
                buf = pending_kv_b.setdefault(layer_idx, {})
                buf[which] = arr
                if "k" in buf and "v" in buf:
                    combined = combine_kv_b_3d(buf["k"], buf["v"])
                    del pending_kv_b[layer_idx]
                    emit_pairs = [
                        (f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight", combined)
                    ]
                else:
                    emit_pairs = []  # buffered until the pair arrives
            elif name.endswith(".exp_probs_b.bias"):
                # noaux_tc gate correction bias: gguf2mlx matches rest=="exp_probs_b"
                # but actual GGUF tensor is "blk.N.exp_probs_b.bias". Rename directly.
                m_ep = re.match(r"blk\.(\d+)\.exp_probs_b\.bias$", name)
                if m_ep:
                    layer_idx = m_ep.group(1)
                    emit_pairs = [
                        (f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", arr)
                    ]
                else:
                    emit_pairs = _plan_tensor_emit(name, arr, arch, mla_dims, pending_kv_b)
            else:
                # Standard path: name mapping + MLA transforms (expert split, etc.)
                emit_pairs = _plan_tensor_emit(
                    name, arr, arch, mla_dims, pending_kv_b
                )

            for hf_name, out_arr in emit_pairs:
                rule = quant_rule(hf_name)

                if isinstance(rule, dict) and hf_name.endswith(".weight"):
                    # Quantize via mlx
                    base = hf_name[:-len(".weight")]
                    w_mx = mx.array(out_arr)
                    wq, scales, biases = mx.quantize(
                        w_mx,
                        group_size=rule["group_size"],
                        bits=rule["bits"],
                        mode=rule["mode"],
                    )
                    writer.add(f"{base}.weight", wq)
                    writer.add(f"{base}.scales", scales)
                    writer.add(f"{base}.biases", biases)
                    quantization_block[base] = rule
                    quantized_count += 1
                else:
                    # Keep fp16
                    writer.add(hf_name, mx.array(out_arr).astype(mx.float16))
                    fp16_count += 1

            if tensor_count % 50 == 0:
                print(f"    [{tensor_count}] {tensor_count} tensors processed, "
                      f"{writer.total_bytes / 1e9:.1f} GB out")

    print(f"\n  Processed {tensor_count} tensors → "
          f"{quantized_count} quantized + {fp16_count} fp16 = "
          f"{quantized_count + fp16_count} emitted")

    # ── Step 5: Finalize shards + index ──
    print("\n[4/4] Finalizing shards + index...")
    total_shards = writer.finalize()

    # mlx-lm's deepseek_v32.sanitize() STACKS the per-expert linears
    #   model.layers.N.mlp.experts.E.{gate,up,down}_proj
    # into the single stacked module
    #   model.layers.N.mlp.switch_mlp.{gate,up,down}_proj
    # at load time. nn.quantize()'s class_predicate looks up the STACKED module
    # path in config["quantization"]; if absent it falls back to the default
    # bits (4), which mismatches 2-bit experts → shape error
    #   expected (256,2048,768) got (256,2048,384).
    # Mirror each layer's per-expert rule onto the stacked switch_mlp path.
    _switch_added = 0
    for _k, _v in list(quantization_block.items()):
        _m = re.match(
            r"(model\.layers\.\d+)\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",
            _k,
        )
        if _m and isinstance(_v, dict):
            _sm = f"{_m.group(1)}.mlp.switch_mlp.{_m.group(2)}"
            if _sm not in quantization_block:
                quantization_block[_sm] = _v
                _switch_added += 1
    if _switch_added:
        print(f"    + added {_switch_added} switch_mlp quant entries "
              f"(stacked-expert paths)")

    # Write config.json with quantization block
    config["quantization"] = quantization_block
    config["quantization_config"] = quantization_block
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Write generation_config.json for GLM-5.2
    if arch == "glm-dsa":
        gen_cfg = {
            "_from_model_config": True,
            "eos_token_id": [154820, 154827, 154829],
            "pad_token_id": 154820,
            "temperature": 1.0,
            "top_p": 0.95,
            "transformers_version": "5.12.0",
        }
        with open(output_dir / "generation_config.json", "w") as f:
            json.dump(gen_cfg, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"✅ Conversion complete in {elapsed / 60:.1f} min")
    print(f"  Shards: {total_shards}")
    print(f"  Output: {writer.total_bytes / 1e9:.2f} GB")
    print(f"  Input:  {total_in / 1e9:.2f} GB")
    print(f"  Ratio:  {writer.total_bytes / total_in:.3f}×")
    print(f"{'=' * 60}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming GGUF → MLX mixed-precision converter")
    parser.add_argument("--gguf-dir", required=True, help="Directory containing multi-shard GGUF")
    parser.add_argument("--output", required=True, help="Output MLX directory")
    parser.add_argument("--dry-run", type=int, default=0,
                        help="Process only N tensors (for testing)")
    parser.add_argument("--reference-tokenizer", default=None,
                        help="Path to a known-good tokenizer.json (or a dir "
                             "containing one) to copy over the GGUF-derived "
                             "tokenizer. REQUIRED for GLM-5.2 so special "
                             "<think>/</think>/tool tokens encode correctly.")
    args = parser.parse_args()
    ok = convert_streaming(args.gguf_dir, args.output, dry_run=args.dry_run,
                           reference_tokenizer=args.reference_tokenizer)
    sys.exit(0 if ok else 1)
