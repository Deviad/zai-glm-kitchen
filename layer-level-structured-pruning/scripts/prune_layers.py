#!/usr/bin/env python3
"""Apply a ShortGPT layer-drop plan to a multi-shard GLM-5.2 GGUF model.

Reads a BI plan (JSON with layers_to_drop + renumber_map + block_count_new)
produced by ``scripts/tracing/analyze_bi_scores.py`` and rewrites all 9
shards of the source model into ``--output-dir``. Each shard is processed
independently (mirrors Phase 7b semantics so the existing exclude/rename KV
code path is reused without surprises):

  * For shards 2..9 (data shards): every tensor whose name matches ``blk.N.*``
    where ``N`` is in ``layers_to_drop`` is EXCLUDED. Every kept ``blk.N.*``
    tensor is RENAMED to ``blk.{new}.*`` per the plan's renumber_map. Non-blk
    tensors (``token_embd.weight`` / ``output.weight`` / ``blk.78.*`` MTP -
    wait, ``blk.78.*`` stays one of the dropped set ONLY if the plan says so;
    in ShortGPT we leave MTP untouched and renumber it after the last kept
    normal layer) are written unchanged.
  * For shard 1 (metadata-only): all KV is copied; ``glm-dsa.block_count`` is
    patched to ``block_count_new`` and ``split.tensors.count`` is patched to
    ``new_tensor_count`` (computed by scanning source shards for tensors
    matching each dropped layer and summing them).

Usage::

    python3 scripts/prune_layers.py \\
        --input-dir  /Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-shortgpt-IQ2S-experts-IQ4NL-rest \\
        --output-dir /Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-shortgpt-pruned-IQ2S-experts-IQ4NL-rest \\
        --plan       reports/glm52_shortgpt_bi_scores.json

The output-dir basename is preserved across shards so the loader still
discovers them via the 0000N-of-00009 pattern. The existing model stays
untouched.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Local import — same folder.
sys.path.insert(0, str(Path(__file__).parent))
import prune_gguf  # noqa: E402

import gguf  # noqa: E402

BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def make_remap(renumber_map: dict[int, int], drop_set: set[int]):
    """Build a callable (orig_name -> str | None) used as prune_gguf's hook."""

    def remap(orig: str) -> str | None:
        m = BLK_RE.match(orig)
        if not m:
            # Non-blk tensor (token_embd.weight / output.weight / norm weights
            # etc). Pass through unchanged.
            return orig
        n = int(m.group(1))
        if n in drop_set:
            return None  # excluded
        # n should always be in renumber_map (the map covers all kept indices).
        if n not in renumber_map:
            # Defensive: an unexpected blk.N appears — pass through unchanged
            # and let the caller see whether the loader complains about a
            # stale index outside the new block_count range.
            return orig
        return f"blk.{renumber_map[n]}.{m.group(2)}"

    return remap


def count_dropped_tensors(input_dir: Path, drop_set: set[int]) -> int:
    """Scan all shards and sum tensors whose name matches a dropped layer.

    Used to patch split.tensors.count in the metadata shard (shard 1).
    """
    total = 0
    per_layer_counts: dict[int, int] = {n: 0 for n in drop_set}
    for shard in sorted(input_dir.glob("*.gguf")):
        r = gguf.GGUFReader(str(shard))
        for rt in r.tensors:
            m = BLK_RE.match(rt.name)
            if not m:
                continue
            n = int(m.group(1))
            if n in drop_set:
                total += 1
                per_layer_counts[n] += 1
    return total, per_layer_counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", "-i", required=True, type=Path)
    ap.add_argument("--output-dir", "-o", required=True, type=Path)
    ap.add_argument("--plan", "-p", required=True, type=Path)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite output shards if they exist")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: input-dir not found: {args.input_dir}", file=sys.stderr)
        return 2
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.force:
            print(f"ERROR: output-dir exists and is non-empty: {args.output_dir} "
                  f"(use --force to overwrite)", file=sys.stderr)
            return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.plan) as f:
        plan = json.load(f)

    drop_set = set(plan["layers_to_drop"])
    renumber_map = {int(k): int(v) for k, v in plan["renumber_map"].items()}
    block_count_new = int(plan["block_count_new"])
    print("=== ShortGPT layer-drop plan ===")
    print(f"  layers to drop:         {sorted(drop_set)}  (n={len(drop_set)})")
    print(f"  kept normal layers:     {len(renumber_map)}  (with layer 0 always kept)")
    print(f"  new glm-dsa.block_count: {block_count_new}")

    # Scan shards first to compute the patched split.tensors.count.
    # Treat MTP / blk.78 specially: if it's not in drop_set (we never drop MTP
    # in ShortGPT), it gets renumbered to blk.{len(kept_normal_layers)}.
    n_dropped_tensors, per_layer = count_dropped_tensors(args.input_dir, drop_set)
    print(f"  total tensors to drop:   {n_dropped_tensors}")
    print("  per-layer dropped:       "
          + ", ".join(f"blk.{n}={c}" for n, c in sorted(per_layer.items())))

    # Original tensor count
    src_shards = sorted(args.input_dir.glob("*.gguf"))
    orig_total = 0
    for sh in src_shards:
        r = gguf.GGUFReader(str(sh))
        orig_total += len(r.tensors)
    new_total_tensor_count = orig_total - n_dropped_tensors
    print(f"  split.tensors.count:   {orig_total} -> {new_total_tensor_count}")

    if "78" not in {str(n) for n in drop_set} and 78 not in drop_set:
        # MTP layer stays. Compute its new index.
        mtp_new_idx = len(renumber_map)  # = count of kept normal layers = 62 with 16 dropped
        print(f"  MTP (blk.78) stays, renumbered -> blk.{mtp_new_idx}")
        renumber_map[78] = mtp_new_idx

    remap = make_remap(renumber_map, drop_set)

    # Replace basename "shortgpt" with "shortgpt-pruned" in each output shard
    # name so we never overwrite the source model by accident.
    src_basename = args.input_dir.name  # e.g. GLM-5.2-shortgpt-IQ2S-experts-IQ4NL-rest
    new_basename = args.output_dir.name
    # The shard filename rewriter just substitutes the input-dir name with
    # the output-dir name. (e.g. GLM-5.2-shortgpt-IQ2S-... -> GLM-5.2-shortgpt-
    # pruned-IQ2S-... in both dir and shard filenames.) No 'shortgpt' string-
    # handling needed; the substitution is purely basename-based.
    src_marker = src_basename
    new_marker = new_basename

    for shard in src_shards:
        out_name = shard.name.replace(src_marker, new_marker)
        if out_name == shard.name:
            # No marker substitution happened (shouldn't happen for our setup,
            # but defensively).
            out_name = shard.name.replace(args.input_dir.name, args.output_dir.name)
        out_path = args.output_dir / out_name
        if out_path.exists() and not args.force:
            print(f"ERROR: output exists: {out_path}", file=sys.stderr)
            return 2
        r = gguf.GGUFReader(str(shard))
        is_meta_only = len(r.tensors) == 0
        print(f"\n== {shard.name} -> {out_name}  "
              f"{'(metadata-only)' if is_meta_only else ''}  "
              f"size={shard.stat().st_size / 1e9:.2f} GB")

        kv_overrides = None
        if is_meta_only:
            # Shard 1: patch block_count + split.tensors.count (everything else
            # passes through unfaded).
            kv_overrides = {
                "glm-dsa.block_count": (
                    block_count_new,
                    gguf.GGUFValueType.UINT32,
                ),
                "split.tensors.count": (
                    new_total_tensor_count,
                    gguf.GGUFValueType.INT32,  # IMPORTANT: must be INT32 (type 5)
                ),
            }
        n_total, n_kept = prune_gguf.prune_gguf(
            shard,
            out_path,
            exclude_patterns=[],  # drop handled by remap returning None
            tensor_name_remap=remap,
            kv_overrides=kv_overrides,
        )
        print(f"   tensors kept/dropped: {n_kept}/{n_total - n_kept}")

    print("\n=== done. Verify with:")
    print(f"  llama-cli -m {args.output_dir}/GLM-5.2-*shortgpt*-00001-of-00009.gguf -p 'Hello' -n 8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
