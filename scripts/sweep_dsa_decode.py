#!/usr/bin/env python3
"""
Context-length sweep: DSA decode tok/s vs context size (default index_topk=2048).

Takes prefix slices of a long prompt at {2K,4K,8K,16K,32K} tokens, appends a
short question, sends to /completion (raw, no chat-template parser), records
decode tok/s from the server's timings block.

Reuses /tmp/longctx_prompt.txt (53_633 tokens, ~5.8 chars/tok) by default;
override with --src.
"""
import json, time, urllib.request, sys, os, argparse

URL = "http://127.0.0.1:8081/completion"
DEFAULT_SRC = "/tmp/longctx_prompt.txt"
TARGETS = [2000, 4000, 8000, 16000, 32000]
CHARS_PER_TOK = 5.8
N_PREDICT = 48


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help=f"source prompt file (default: {DEFAULT_SRC})")
    ap.add_argument("--url", default=URL, help=f"completion endpoint (default: {URL})")
    args = ap.parse_args()

    SRC = args.src
    if not os.path.exists(SRC):
        sys.exit(f"source prompt not found: {SRC} (pass --src)")

    text = open(SRC, encoding="utf-8").read()
    total_tokens_est = int(len(text) / CHARS_PER_TOK)
    # NOTE: char-based truncation (CHARS_PER_TOK=5.8 is an English average) is an
    # APPROXIMATION. CJK / emoji content maps ~1.3-1.5 chars/tok, so target token
    # counts may be off by ~30% for non-English prompts. The server's reported
    # `prompt_n` (actual tokenized count) is the ground truth recorded below.
    # Full tokenizer-aware truncation is deferred; see REMEDIATION_PLAN P2.
    print(f"# source prompt: {SRC}")
    print(f"# source prompt: {len(text)} chars ~= {total_tokens_est} tokens (char-based est)")
    print(f"# sweep targets: {TARGETS} tokens, n_predict={N_PREDICT}")
    print(f"# {'ctx_tgt':>8} {'prompt_n':>9} {'prompt_t/s':>11} {'decode_n':>9} {'decode_t/s':>11}")
    print(f"# {'-'*8} {'-'*9} {'-'*11} {'-'*9} {'-'*11}")

    results = []
    for tgt in TARGETS:
        char_len = int(tgt * CHARS_PER_TOK)
        char_len = min(char_len, len(text) - 200)
        prompt = text[:char_len] + "\n\nQuestion: What is the hidden passphrase mentioned above? Answer in one short sentence:"
        req = {
            "prompt": prompt,
            "n_predict": N_PREDICT,
            "temperature": 0.1,
            "cache_prompt": False,
            "stream": False,
        }
        body = json.dumps(req).encode()
        t0 = time.time()
        try:
            r = urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=1800) as resp:
                raw = resp.read()
            wall = time.time() - t0
            d = json.loads(raw)
            tk = d.get("timings", {})
            pn = tk.get("prompt_n", 0); pps = tk.get("prompt_per_second", 0)
            dn = tk.get("predicted_n", 0); dps = tk.get("predicted_per_second", 0)
            print(f"  {tgt:>8} {pn:>9} {pps:>11.1f} {dn:>9} {dps:>11.2f}   (wall {wall:.0f}s)")
            sys.stdout.flush()
            results.append({"ctx_tgt": tgt, "prompt_n": pn, "prompt_tps": round(pps,2),
                            "decode_n": dn, "decode_tps": round(dps,2), "wall_s": round(wall)})
        except Exception as e:
            print(f"  {tgt:>8}  ERROR: {e}")
            results.append({"ctx_tgt": tgt, "error": str(e)})

    # write machine-readable
    out = "/Volumes/Data NVME/GLM-5.2-kitchen/logs/dsa_sweep_results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(json.dumps(results, indent=2))
    print(f"\n# results -> {out}")


if __name__ == "__main__":
    main()
