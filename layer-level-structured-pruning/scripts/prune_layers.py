#!/usr/bin/env python3
"""Apply a ShortGPT layer-drop plan to a multi-shard GLM-5.2 GGUF model.

Reads a BI plan (JSON with layers_to_drop + renumber_map + block_count_new)
produced by ``layer-level-structured-pruning/scripts/analyze_bi_scores.py`` and rewrites all 9
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

    python3 layer-level-structured-pruning/scripts/prune_layers.py \\
        --input-dir  /Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-shortgpt-IQ2S-experts-IQ4NL-rest \\
        --output-dir /Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-shortgpt-pruned-IQ2S-experts-IQ4NL-rest \\
        --plan       layer-level-structured-pruning/reports/glm52_shortgpt_bi_scores.json

    # Dry-run: validate inputs, scan shards, report changes without writing:
    python3 layer-level-structured-pruning/scripts/prune_layers.py \\
        --input-dir  /Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-shortgpt-IQ2S-experts-IQ4NL-rest \\
        --plan       layer-level-structured-pruning/reports/glm52_shortgpt_bi_scores.json \\
        --dry-run

The output-dir basename is preserved across shards so the loader still
discovers them via the 0000N-of-00009 pattern. The existing model stays
untouched.
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path

# Local import — same folder. Keep eager so --dry-run also validates the
# pruning stack imports as a lightweight integration check.
sys.path.insert(0, str(Path(__file__).parent))
try:
    import prune_gguf  # noqa: E402
except Exception as exc:  # pragma: no cover - import-time integration guard
    print(f"ERROR: failed to import prune_gguf integration dependency: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    import gguf  # noqa: E402
except Exception as exc:  # pragma: no cover - import-time integration guard
    print(f"ERROR: failed to import gguf dependency: {exc}", file=sys.stderr)
    sys.exit(1)

BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def validate_prune_gguf_api() -> None:
    """Validate prune_gguf hooks needed by both dry-run and normal mode."""
    if not hasattr(prune_gguf, "prune_gguf"):
        raise AttributeError("module prune_gguf has no prune_gguf()")
    params = inspect.signature(prune_gguf.prune_gguf).parameters
    required = {"tensor_name_remap", "kv_overrides"}
    missing = sorted(required.difference(params))
    if missing:
        raise TypeError(f"prune_gguf.prune_gguf() missing required hook parameter(s): {missing}")


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


def count_dropped_tensors(input_dir: Path, drop_set: set[int]) -> tuple[int, dict[int, int]]:
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
    ap.add_argument("--output-dir", "-o", type=Path,
                    help="Output directory for pruned shards (required without --dry-run)")
    ap.add_argument("--plan", "-p", required=True, type=Path)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite output shards if they exist")
    ap.add_argument("--dry-run", "-n", action="store_true",
                    help="Validate inputs, scan shards, report planned changes, "
                         "and exit without writing any files")
    args = ap.parse_args()

    try:
        validate_prune_gguf_api()
    except Exception as exc:
        print(f"ERROR: prune_gguf integration check failed: {exc}", file=sys.stderr)
        return 1

    if not args.input_dir.is_dir():
        print(f"ERROR: input-dir not found: {args.input_dir}", file=sys.stderr)
        return 1 if args.dry_run else 2

    # ── Dry-run mode ──────────────────────────────────────────────────
    if args.dry_run:
        print("=" * 44)
        print("  ShortGPT layer-drop -- DRY RUN")
        print("=" * 44)
        print()
        print(f"  Input model : {args.input_dir}")
        print(f"  BI plan     : {args.plan}")
        print()

        # Validate plan
        if not args.plan.is_file():
            print(f"ERROR: plan not found: {args.plan}", file=sys.stderr)
            return 1
        try:
            with open(args.plan) as f:
                plan = json.load(f)
            required_keys = {"layers_to_drop", "renumber_map", "block_count_new"}
            missing = sorted(required_keys.difference(plan))
            if missing:
                raise KeyError(f"missing required plan key(s): {missing}")
            drop_set = set(plan["layers_to_drop"])
            renumber_map = {int(k): int(v) for k, v in plan["renumber_map"].items()}
            block_count_new = int(plan["block_count_new"])
        except Exception as exc:
            print(f"ERROR: failed to load/validate dry-run plan {args.plan}: {exc}", file=sys.stderr)
            return 1
        print("--- BI plan (loaded) ---")
        print(f"  layers to drop:         {sorted(drop_set)}  (n={len(drop_set)})")
        print(f"  kept normal layers:     {len(renumber_map)}")
        print(f"  new block_count:        {block_count_new}")
        print()

        # Scan shards
        try:
            n_dropped_tensors, per_layer = count_dropped_tensors(args.input_dir, drop_set)
            src_shards = sorted(args.input_dir.glob("*.gguf"))
            if not src_shards:
                raise FileNotFoundError(f"no GGUF shards found under {args.input_dir}")
            total_tensors = 0
            all_tensor_types: dict[str, int] = {}
            for sh in src_shards:
                r = gguf.GGUFReader(str(sh))
                total_tensors += len(r.tensors)
                for rt in r.tensors:
                    try:
                        tname = gguf.GGMLQuantizationType(rt.tensor_type).name
                    except Exception:
                        tname = f"type={int(rt.tensor_type)}"
                    all_tensor_types[tname] = all_tensor_types.get(tname, 0) + 1
        except Exception as exc:
            print(f"ERROR: dry-run GGUF shard scan failed: {exc}", file=sys.stderr)
            return 1

        new_total_tensor_count = total_tensors - n_dropped_tensors
        print(f"  Total tensors across all shards: {total_tensors}")
        print(f"  Tensors to drop:                 {n_dropped_tensors}")
        print(f"  Tensors to keep:                 {total_tensors - n_dropped_tensors}")
        print()

        # Per-layer drop details
        print("  Per-layer drop details:")
        for layer in sorted(per_layer):
            count = per_layer[layer]
            if count > 0:
                print(f"    blk.{layer:2d}: {count} tensors")
        print()

        print(f"  split.tensors.count: {total_tensors} -> {new_total_tensor_count}")
        print()

        # MTP handling
        if "78" not in {str(n) for n in drop_set} and 78 not in drop_set:
            mtp_new_idx = len(renumber_map)
            print(f"  MTP blk.78 stays -> renumbered to blk.{mtp_new_idx}")
        print()

        # Example renumber map
        print("  Example renumber map (first 5 kept layers):")
        for old in sorted(renumber_map.keys())[:5]:
            print(f"    blk.{old:2d} -> blk.{renumber_map[old]:2d}")
        if len(renumber_map) > 5:
            print(f"    ... ({len(renumber_map) - 5} more)")
        print()

        # Shard tensor types
        if all_tensor_types:
            print("  Current shard tensor types:")
            for k, v in sorted(all_tensor_types.items(), key=lambda x: -x[1]):
                print(f"    {v:>6d}  {k}")
            print()

        print("--- Summary ---")
        print("  Status: DRY RUN COMPLETE -- no files written, no shards updated")
        print("  Next step: run prune_layers.py without --dry-run to apply changes")
        print("=" * 44)
        return 0

    # ── Normal mode: write pruned shards ──────────────────────────────
    if not args.output_dir:
        print("ERROR: --output-dir is required without --dry-run", file=sys.stderr)
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
    print()

    # Scan shards first to compute the patched split.tensors.count.
    # Treat MTP / blk.78 specially: if it's not in drop_set (we never drop MTP
    # in ShortGPT), it gets renumbered to blk.{len(kept_normal_layers)}.
    n_dropped_tensors, per_layer = count_dropped_tensors(args.input_dir, drop_set)
    print(f"  total tensors to drop:   {n_dropped_tensors}")
    print("  per-layer dropped:       "
          + ", ".join(f"blk.{n}={c}" for n, c in sorted(per_layer.items())))

    # Original tensor count
    src_shards_orig = sorted(args.input_dir.glob("*.gguf"))
    orig_total = 0
    for sh in src_shards_orig:
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

    for shard in src_shards_orig:
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
