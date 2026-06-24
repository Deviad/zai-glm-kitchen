#!/usr/bin/env python3
"""
2-query prompt-cache test on shortgpt-pruned GGUF (server already running on :8081).

Q1: cold prefill of the 53.6K-token needle prompt + "what is the passphrase?"
Q2: SAME prompt prefix + different question (rotation interval) -> cache hit.

Prints wall time + server-reported timings (prompt_n, prompt_per_second,
predicted_n, predicted_per_second) for each, plus answers.
"""
import json, time, urllib.request, sys

URL = "http://127.0.0.1:8081/v1/chat/completions"
SRC = "/tmp/longctx_prompt.txt"

text = open(SRC, encoding="utf-8").read()
print(f"# prompt file: {len(text)} chars (~{int(len(text)/5.8)} tokens est)")

Q1 = "What is the hidden passphrase mentioned above? Answer in one short sentence."
Q2 = "How often must the passphrase be rotated? Answer in one short sentence."

def ask(question, label):
    body = {
        "messages": [
            {"role": "user", "content": text + "\n\nQuestion: " + question}
        ],
        "max_tokens": 64,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    t0 = time.time()
    req = urllib.request.Request(URL, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        raw = resp.read()
    wall = time.time() - t0
    d = json.loads(raw)
    tk = d.get("timings", {})
    ans = d["choices"][0]["message"]["content"]
    print(f"\n=== {label} ===")
    print(f"  wall:           {wall:.1f}s")
    print(f"  prompt_n:       {tk.get('prompt_n','?')}")
    print(f"  prompt t/s:     {tk.get('prompt_per_second','?'):.1f}"
          if isinstance(tk.get('prompt_per_second'), (int,float)) else f"  prompt t/s: ?")
    print(f"  predicted_n:    {tk.get('predicted_n','?')}")
    print(f"  predicted t/s:  {tk.get('predicted_per_second','?'):.2f}"
          if isinstance(tk.get('predicted_per_second'), (int,float)) else f"  predicted t/s: ?")
    print(f"  cached_tokens:  {tk.get('prompt_n_cached','?')}  (cache hit indicator)")
    print(f"  answer: {ans!r}")
    return {"wall": wall, "timings": tk, "answer": ans}

r1 = ask(Q1, "Q1: COLD prefill (passphrase)")
r2 = ask(Q2, "Q2: WARM (cache hit, rotation)")

print("\n=== SUMMARY ===")
p1 = r1["timings"].get("prompt_per_second", 0) or 0
p2 = r2["timings"].get("prompt_per_second", 0) or 0
print(f"  Q1 wall={r1['wall']:.1f}s  prompt_t/s={p1:.1f}")
print(f"  Q2 wall={r2['wall']:.1f}s  prompt_t/s={p2:.1f}")
if r1["wall"] > 0:
    print(f"  speedup Q2/Q1 wall: {r1['wall']/max(r2['wall'],0.01):.1f}x")
