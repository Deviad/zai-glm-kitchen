#!/usr/bin/env python3
"""Prune tensors matching a name pattern from a GGUF file (shard-level).

Reads an input GGUF, copies KV metadata + tensors whose name does NOT match
the exclude glob pattern, writes a new (smaller) GGUF.

Example (drop MTP layer from a single shard):

    python scripts/prune_gguf.py \\
        --input  /path/GLM-5.2-mixed-00009-of-00009.gguf \\
        --output /path/GLM-5.2-pruned-00009-of-00009.gguf \\
        --exclude 'blk.78.*'

Verify via `llama-cli --model /path/GLM-5.2-pruned-00001-of-00009.gguf`
pointing at the new (pruned shard 9) + unchanged shards 1-8. The loader's
`TENSOR_SKIP | TENSOR_NOT_REQUIRED` path for blk.78 tolerates their absence.

Non-destructive: input file is left untouched; output is a new file.
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

import numpy as np

import gguf


def _field_to_kv_value(field: gguf.ReaderField):
    """Read a GGUFReader field's value in a form suitable for GGUFWriter.add_key_value."""
    value = field.contents()
    vtype = field.types[0] if field.types else gguf.GGUFValueType.STRING
    return value, vtype, field.types[-1] if len(field.types) > 1 else None


def _build_writer_key(writer: gguf.GGUFWriter, key: str, value, vtype, sub_type) -> None:
    """Add a KV pair to the writer, using the correct adder for the type."""
    # Always set on kv_data[0]; add_key_value handles overwrite warning.
    if vtype == gguf.GGUFValueType.ARRAY:
        arr = list(value)
        writer.add_array(key, arr)
        return
    # Scalar types — dispatch by GGUFValueType name.
    adder_map = {
        gguf.GGUFValueType.UINT8:   writer.add_uint8,
        gguf.GGUFValueType.INT8:    writer.add_int8,
        gguf.GGUFValueType.UINT16:  writer.add_uint16,
        gguf.GGUFValueType.INT16:   writer.add_int16,
        gguf.GGUFValueType.UINT32:  writer.add_uint32,
        gguf.GGUFValueType.INT32:   writer.add_int32,
        gguf.GGUFValueType.UINT64:  writer.add_uint64,
        gguf.GGUFValueType.INT64:   writer.add_int64,
        gguf.GGUFValueType.FLOAT32: writer.add_float32,
        gguf.GGUFValueType.FLOAT64: writer.add_float64,
        gguf.GGUFValueType.BOOL:    writer.add_bool,
        gguf.GGUFValueType.STRING: lambda k, v: writer.add_string(k, v),
    }
    adder = adder_map.get(vtype)
    if adder is None:
        raise NotImplementedError(f"KV pair {key!r} has unsupported type {vtype}")
    # `general.architecture` is special — GGUFWriter.add_architecture takes no arg.
    if key == "general.architecture":
        writer.add_architecture()
        return
    # `general.file_type` is mapped via add_file_type.
    if key == "general.file_type":
        writer.add_file_type(int(value))
        return
    if key == "general.quantization_version":
        writer.add_quantization_version(int(value))
        return
    adder(key, value)


def prune_gguf(input_path: Path, output_path: Path, exclude_patterns: list[str],
                *,
                tensor_name_remap=None,
                kv_overrides: dict | None = None) -> tuple[int, int]:
    """Prune tensors matching any of exclude_patterns from input_path -> output_path.

    Optional hooks:
      * tensor_name_remap: callable(orig_name: str) -> str | None. If it returns
        None, the tensor is excluded. If it returns a string, that string is
        used as the new tensor name (so callers can rename e.g. blk.78 → blk.62
        when layer indices are renumbered by ShortGPT pruning).
      * kv_overrides: dict mapping KV name -> (value, GGUFValueType) tuples. The
        writer uses these values instead of the source shard's values (so callers
        can patch e.g. glm-dsa.block_count and split.tensors.count in-place).

    Returns (n_total, n_kept).
    """
    print(f"== Pruning GGUF shard: {input_path.name}")
    print(f"   Input size:          {input_path.stat().st_size / 1e9:.3f} GB")
    print(f"   Exclude pattern(s):  {exclude_patterns}")

    reader = gguf.GGUFReader(str(input_path))
    arch_field = reader.fields.get("general.architecture")
    arch_value = arch_field.contents() if arch_field else "llama"
    print(f"   Architecture:        {arch_value}")

    # For data-only shards (shards 2..N), the architecture KV is absent and
    # writer.add_architecture would write a spurious one. Use the writer ctor
    # with the read arch — for data shards it falls back to "llama" which
    # is acceptable since gguf-py doesn't enforce arch-tensor matching.
    writer = gguf.GGUFWriter(str(output_path), arch_value, endianess=gguf.GGUFEndian.LITTLE)

    # Copy KV metadata (everything except architecture + GGUF.* pseudo fields)
    n_kv_copied = 0
    n_kv_skipped = 0
    for fname, field in reader.fields.items():
        if fname == "general.architecture":
            # Already done via GGUFWriter(arch_value)
            continue
        if fname.startswith("GGUF."):
            # Pseudo-meta fields injected by the reader; writer adds them
            # automatically via write_header_to_file.
            continue
        try:
            value, vtype, sub_type = _field_to_kv_value(field)
        except Exception as e:
            print(f"   WARN: skipping KV field {fname!r} (parse failed: {e})")
            n_kv_skipped += 1
            continue
        # Caller-supplied KV overrides take precedence — let them patch
        # values like glm-dsa.block_count without rewriting the source's KV.
        if kv_overrides is not None and fname in kv_overrides:
            value, vtype = kv_overrides[fname]
        try:
            _build_writer_key(writer, fname, value, vtype, sub_type)
        except (NotImplementedError, ValueError) as e:
            print(f"   WARN: skipping KV field {fname!r} (write failed: {e})")
            n_kv_skipped += 1
            continue
        n_kv_copied += 1
    print(f"   KV copied/skip:      {n_kv_copied} copied, {n_kv_skipped} skipped")

    # Copy tensors (skipping those matching exclude_patterns)
    n_total = len(reader.tensors)
    n_kept = 0
    n_dropped_bytes = 0
    n_seen_bytes = 0
    for rt in reader.tensors:
        n_seen_bytes += rt.n_bytes
        if any(fnmatch.fnmatch(rt.name, pat) for pat in exclude_patterns):
            n_dropped_bytes += rt.n_bytes
            print(f"   drop: {rt.name}  ({rt.n_bytes / 1e6:.2f} MB)")
            continue
        # Optional caller-supplied name remap: lets ShortGPT layer-drop callers
        # rename blk.{orig}.* -> blk.{new}.* as the kept tensors stream out.
        out_name = rt.name
        if tensor_name_remap is not None:
            remapped = tensor_name_remap(rt.name)
            if remapped is None:
                n_dropped_bytes += rt.n_bytes
                print(f"   drop (via remap): {rt.name}  ({rt.n_bytes / 1e6:.2f} MB)")
                continue
            if remapped != rt.name:
                print(f"   rename: {rt.name} -> {remapped}  ({rt.n_bytes / 1e6:.2f} MB)")
            out_name = remapped
        raw_dtype = gguf.GGMLQuantizationType(rt.tensor_type)
        # F32/F16/F64 tensors: gguf-py reader exposes rt.data as float32/16/64
        # numpy array with shape = element_shape reversed (C-contiguous order).
        # Pass raw_shape = rt.data.shape (numpy order) so the writer can reverse
        # it back to the on-disk dim order during write_ti_data_to_file.
        # Quantized tensors: rt.data is a uint8 memmap with shape =
        # quant_shape[0:-1] + (quant_shape[-1] * type_size). Same reversal rule
        # — pass rt.data.shape so writer can reverse it.
        if rt.tensor_type in (0, 1, 2):  # F32, F16, F64
            tensor = np.array(rt.data)  # copy out of mmap
            writer.add_tensor(out_name, tensor, raw_shape=list(int(d) for d in rt.data.shape),
                              raw_dtype=raw_dtype)
            n_kept += 1
        else:
            raw = np.frombuffer(bytes(rt.data), dtype=np.uint8)
            writer.add_tensor(out_name, raw, raw_shape=list(int(d) for d in rt.data.shape),
                              raw_dtype=raw_dtype)
            n_kept += 1
        if n_kept % 25 == 0:
            print(f"   ... {n_kept} tensors queued ({n_total - n_kept} dropped)")

    print(f"   Tensor totals:       {n_kept}/{n_total} kept ({n_total - n_kept} dropped)")

    # Write the new GGUF. Always use temp-file mode so multi-GB writes don't
    # blow up memory.
    writer.use_temp_file = True
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    output_size = output_path.stat().st_size
    print(f"   Output size:         {output_size / 1e9:.3f} GB")
    print(f"   Saved:               {(input_path.stat().st_size - output_size) / 1e9:.3f} GB "
          f"({(1 - output_size / input_path.stat().st_size) * 100:.2f}%)")
    return n_total, n_kept


def patch_split_tensor_count(input_path: Path, output_path: Path, new_count: int) -> None:
    """Copy a metadata-only shard (shard 1) with split.tensors.count patched.

    Used after pruning tensors from data shards: the cross-shard tensor
    total stored in shard 1's KV must match the actual count of tensors
    found across all shards (else llama_model_loader asserts).
    """
    print(f"== Patching shard 1 split.tensors.count: {input_path.name}")
    print(f"   Output:  {output_path}")
    print(f"   New count: {new_count}")

    # Delegate to prune_gguf with an empty exclude list, then post-patch the KV.
    # Simpler: copy ALL KV (excluding GGUF.* pseudo fields) and tensors,
    # overriding split.tensors.count.
    reader = gguf.GGUFReader(str(input_path))
    arch_field = reader.fields.get("general.architecture")
    arch_value = arch_field.contents() if arch_field else "llama"
    writer = gguf.GGUFWriter(str(output_path), arch_value, endianess=gguf.GGUFEndian.LITTLE)

    n_kv_copied = 0
    n_kv_skipped = 0
    for fname, field in reader.fields.items():
        if fname == "general.architecture":
            continue
        if fname.startswith("GGUF."):
            continue
        if fname == "split.tensors.count":
            # llama.cpp loader asserts this KV is INT32 (type 5), so use
            # add_int32 — even though py-gguf's GGUFValueType would pick
            # UINT32 (type 4) by default. Mismatched types cause a load-time
            # rejection in llama_model_loader.cpp:
            #   "key split.tensors.count has wrong type u32 but expected type i32"
            writer.add_int32(fname, new_count)
            n_kv_copied += 1
            continue
        try:
            value, vtype, sub_type = _field_to_kv_value(field)
            _build_writer_key(writer, fname, value, vtype, sub_type)
            n_kv_copied += 1
        except Exception as e:
            print(f"   WARN: skipping KV field {fname!r} (failed: {e})")
            n_kv_skipped += 1
    print(f"   KV copied/skip: {n_kv_copied} copied, {n_kv_skipped} skipped")

    # Shard 1 has 0 tensors — no tensor copy, but still write a valid GGUF.
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    print(f"   Output size: {output_path.stat().st_size / 1e6:.2f} MB")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, type=Path, help="Input GGUF shard path")
    ap.add_argument("--output", "-o", required=True, type=Path, help="Output GGUF shard path")
    ap.add_argument("--exclude", "-e", action="append", default=[],
                    help="Tensor-name glob to exclude (may be repeated, e.g. 'blk.78.*')")
    ap.add_argument("--patch-shard1-split-count", type=int, default=None,
                    help="Override split.tensors.count KV value (used for metadata-only shard 1 "
                         "after pruning tensors from data shards). When set, ignores --exclude.")
    ap.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2
    if args.output.exists() and not args.force:
        print(f"ERROR: output exists: {args.output} (use --force to overwrite)", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.patch_shard1_split_count is not None:
        patch_split_tensor_count(args.input, args.output, args.patch_shard1_split_count)
        return 0

    if not args.exclude:
        print("ERROR: --exclude is required unless --patch-shard1-split-count is set", file=sys.stderr)
        return 2

    prune_gguf(args.input, args.output, args.exclude)
    return 0


if __name__ == "__main__":
    sys.exit(main())
