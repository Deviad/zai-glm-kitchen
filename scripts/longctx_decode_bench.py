#!/usr/bin/env python3
"""
Long-context decode benchmark: dense vs sparse-gather DSA attention.

Sends two queries against a server already running on :8081:
  Q1: cold prefill of the 53K-token needle prompt + question (thinking ON)
  Q2: SAME prompt prefix + different question -> cache hit, warm decode

The Q2 `predicted_per_second` (decode tok/s on a warm cache) is the headline
metric comparing the two attention paths. Thinking is ON so the model
produces coherent long-context output (thinking leaked into content, but
the decode speed is still a valid perf signal).

Usage:
  python scripts/longctx_decode_bench.py <label> <out_json>
  e.g. python scripts/longctx_decode_bench.py sparse /tmp/bench_sparse.json
"""
import json, time, urllib.request, sys, os

URL = "http://127.0.0.1:8081/v1/chat/completions"
SRC = os.environ.get("LONGCTX_PROMPT", "/tmp/longctx_prompt.txt")

text = open(SRC, encoding="utf-8").read()
print(f"# prompt file: {SRC} ({len(text)} chars, ~{int(len(text)/5.8)} tokens est)")

Q1 = "What is the hidden passphrase mentioned above? Answer in one short sentence."
Q2 = "How often must the passphrase be rotated? Answer in one short sentence."

def ask(question, label, max_tokens=384):
    body = {
        "messages": [{"role": "user", "content": text + "\n\nQuestion: " + question}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    data = json.dumps(body).encode()
    t0 = time.time()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            raw = resp.read()
    except Exception as e:
        wall = time.time() - t0
        print(f"!! {label} failed after {wall:.1f}s: {e}")
        return {"wall": wall, "error": str(e)}
    wall = time.time() - t0
    d = json.loads(raw)
    tk = d.get("timings", {})
    msg = d.get("choices", [{}])[0].get("message", {})
    ans = msg.get("content", "")
    rc = msg.get("reasoning_content", "")
    print(f"\n=== {label} ===")
    print(f"  wall:           {wall:.1f}s")
    if isinstance(tk.get("prompt_per_second"), (int, float)):
        print(f"  prompt t/s:     {tk['prompt_per_second']:.1f}")
    print(f"  prompt_n:       {tk.get('prompt_n','?')}  cached: {tk.get('prompt_n_cached','?')}")
    if isinstance(tk.get("predicted_per_second"), (int, float)):
        print(f"  predicted t/s: {tk['predicted_per_second']:.2f}")
    print(f"  predicted_n:    {tk.get('predicted_n','?')}")
    print(f"  answer: {ans!r}")
    if rc:
        print(f"  reasoning[:120]: {rc[:120]!r}")
    return {"wall": wall, "timings": tk, "answer": ans, "reasoning_head": rc[:200]}

label = sys.argv[1] if len(sys.argv) > 1 else "run"
out_path = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/bench_{label}.json"

r1 = ask(Q1, f"{label} Q1: COLD prefill (passphrase)")
r2 = ask(Q2, f"{label} Q2: WARM decode (rotation)")

summary = {"label": label, "q1": r1, "q2": r2}
if "timings" in r2 and isinstance(r2["timings"].get("predicted_per_second"), (int, float)):
    print(f"\n=== {label} HEADLINE: warm decode = {r2['timings']['predicted_per_second']:.2f} tok/s ===")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"# wrote {out_path}")
