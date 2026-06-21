#!/usr/bin/env python3
"""Verify a mixed-precision GLM-5.2 GGUF shard: experts at 2-bit, rest at 4-bit.

Usage: uv run --with gguf python verify_glm52_mixed.py <shard1.gguf> [<shard2.gguf> ...]
"""
import sys
from collections import Counter
from pathlib import Path

import gguf

EXPERT_FRAGMENTS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


def classify(name: str) -> str:
    is_expert = any(f in name for f in EXPERT_FRAGMENTS)
    if is_expert and name.startswith("blk.78."):
        return "MTP-EXPERT(4-bit exception)"
    return "EXPERT(2-bit)" if is_expert else "other(4-bit/high)"


def main(shards: list[str]) -> int:
    total_type_counts: Counter[tuple[str, str]] = Counter()
    expert_types: Counter[str] = Counter()
    mtp_expert_types: Counter[str] = Counter()
    other_types: Counter[str] = Counter()
    n_tensors = 0

    for shard in shards:
        r = gguf.GGUFReader(shard)
        arch = None
        f = r.get_field("general.architecture")
        if f is not None:
            arch = f.contents()
        ftype = None
        f = r.get_field("general.file_type")
        if f is not None:
            ftype = int(f.contents().tolist()[0]) if hasattr(f.contents(), "tolist") else f.contents()
        print(f"\n=== {Path(shard).name} ===  arch={arch}  file_type={ftype}  n_tensors={len(r.tensors)}")

        for t in r.tensors:
            n_tensors += 1
            cls = classify(t.name)
            # ggml type enum index
            try:
                tname = gguf.GGMLQuantizationType(t.tensor_type).name
            except Exception:
                tname = f"type={int(t.tensor_type)}"
            total_type_counts[(cls, tname)] += 1
            if cls == "EXPERT(2-bit)":
                expert_types[tname] += 1
            elif cls == "MTP-EXPERT(4-bit exception)":
                mtp_expert_types[tname] += 1
            else:
                other_types[tname] += 1

    print("\n========== SUMMARY ==========")
    print(f"Total tensors scanned: {n_tensors}")
    print("\nExpert tensors (want: IQ2_S / IQ2_XXS / IQ2_M):")
    for k, v in sorted(expert_types.items(), key=lambda x: -x[1]):
        print(f"  {v:6d}  {k}")
    print("\nMTP blk.78 expert tensors (intentional exception; want: IQ4_NL):")
    for k, v in sorted(mtp_expert_types.items(), key=lambda x: -x[1]):
        print(f"  {v:6d}  {k}")
    print("\nOther tensors (want: IQ4_NL/F32/Q8/etc.; no IQ2):")
    for k, v in sorted(other_types.items(), key=lambda x: -x[1]):
        print(f"  {v:6d}  {k}")

    # Hard assertions for the intended mapping.
    bad_experts = {k: v for k, v in expert_types.items() if not k.startswith("IQ2_")}
    if bad_experts:
        print(f"\n❌ FAIL: normal expert tensors not at 2-bit: {bad_experts}")
        return 1
    bad_mtp = {k: v for k, v in mtp_expert_types.items() if k != "IQ4_NL"}
    if bad_mtp:
        print(f"\n❌ FAIL: blk.78 MTP expert exception not at IQ4_NL: {bad_mtp}")
        return 1
    bad_other = {k: v for k, v in other_types.items() if k.startswith("IQ2_")}
    if bad_other:
        print(f"\n❌ FAIL: non-expert tensors unexpectedly at IQ2: {bad_other}")
        return 1
    if not expert_types:
        print("\n⚠ no normal expert tensors found in scanned shards")
    print("\n✓ Mapping verified: normal routed experts are 2-bit; blk.78 MTP experts are IQ4_NL; non-experts are not IQ2.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
