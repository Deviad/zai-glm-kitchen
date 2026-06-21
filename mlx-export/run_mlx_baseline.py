#!/usr/bin/env python
"""MLX baseline: short merge-sort + trivial prompts against the mixed-precision export.

Mirrors scripts/baselines/glm52_merge_sort_baseline.sh (the GGUF baseline)
but runs against the MLX export at /Volumes/Data NVME/GLM-5.2-MLX/...

Usage:
    python mlx-export/run_mlx_baseline.py [--model PATH] [--prompt PROMPT]
    python mlx-export/run_mlx_baseline.py --trivial   # quick 3-prompt sanity

Acceptance: AC6 in PLAN.md §7.M — short merge-sort must produce a correct
(non-degenerate, non-repeating) implementation.
"""
import argparse
import json
import os
import sys
import time

OUT_DEFAULT = "/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-shortgpt-pruned-mixed-mlx"
TRIVIAL_PROMPTS = [
    "What is 2+2?",
    "Say hello.",
    "Complete: The sky is",
]
MERGE_SORT_PROMPT = (
    "Write down a merge sort algorithm (non-recursive) in Python. "
    "Do not explain your reasoning. Output the Python code first, then one short sentence."
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=OUT_DEFAULT,
                   help="Path to the MLX model dir (default: 2-bit export)")
    p.add_argument("--prompt", default=None,
                   help="Custom prompt (overrides --trivial and merge-sort)")
    p.add_argument("--trivial", action="store_true",
                   help="Run the 3 trivial prompts for a fast sanity check")
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()

    from mlx_lm import load, stream_generate

    print(f"Loading model: {args.model}", flush=True)
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    if args.prompt:
        prompts = [args.prompt]
    elif args.trivial:
        prompts = TRIVIAL_PROMPTS
    else:
        prompts = [MERGE_SORT_PROMPT]

    results = []
    for q in prompts:
        messages = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        text = ""
        t0 = time.time()
        n = 0
        for resp in stream_generate(model, tokenizer, prompt, max_tokens=args.max_tokens):
            text += resp.text if hasattr(resp, "text") else str(resp)
            n += 1
        elapsed = time.time() - t0
        tps = n / elapsed if elapsed > 0 else 0
        print(f"\n{'='*60}")
        print(f"PROMPT: {q}")
        print(f"({n} tok, {elapsed:.1f}s, {tps:.1f} tok/s)")
        print(f"OUTPUT: {text[:800]}")
        if len(text) > 800:
            print(f"... ({len(text)} chars total)")
        results.append({"prompt": q, "tokens": n, "elapsed_s": elapsed, "text": text})

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        rep_ratio = _repetition_ratio(r["text"])
        print(f"  {r['prompt'][:40]:<42} | {r['tokens']:4d} tok | {r['elapsed_s']:5.1f}s | rep={rep_ratio:.2f}")
    
    # Save results
    out_path = os.path.join(os.path.dirname(args.model), "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


def _repetition_ratio(text: str) -> float:
    """Rough degeneracy signal: fraction of tokens that are a repeat of the previous."""
    if not text:
        return 0.0
    words = text.split()
    if len(words) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
    return repeats / (len(words) - 1)


if __name__ == "__main__":
    sys.exit(main() or 0)
