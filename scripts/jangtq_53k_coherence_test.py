#!/usr/bin/env python3
"""
JANGTQ_K (MLX) long-context coherence test vs the IQ2S-GGUF gibberish result.

Sends the 53K BLUE-FALCON needle prompt to a vMLX server (port 8082) and checks
whether the response is coherent English (JANGTQ_K passes) or gibberish (same
failure mode as IQ2S GGUF). vMLX returns reasoning_content + content separately
and uses `usage` (not `timings`), so we derive tok/s from completion_tokens/wall.
"""
import json, time, urllib.request, sys, os

URL = "http://127.0.0.1:8082/v1/chat/completions"
MODEL = os.environ.get("VMLX_MODEL", "glm52-jangtq-k")
SRC = os.environ.get("LONGCTX_PROMPT", "/tmp/longctx_prompt.txt")

text = open(SRC, encoding="utf-8").read()
print(f"# prompt file: {SRC} ({len(text)} chars, ~{int(len(text)/5.8)} tokens est)")
print(f"# model: {MODEL}")

Q = "What is the hidden passphrase mentioned above? Answer in one short sentence."
body = {
    "model": MODEL,
    "messages": [{"role": "user", "content": text + "\n\nQuestion: " + Q}],
    "max_tokens": 1024,
    "temperature": 0.1,
}
data = json.dumps(body).encode()
t0 = time.time()
req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=5400) as resp:
        raw = resp.read()
    wall = time.time() - t0
    d = json.loads(raw)
    msg = d["choices"][0]["message"]
    rc = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    usage = d.get("usage", {})
    ptoks = usage.get("prompt_tokens", "?")
    ctoks = usage.get("completion_tokens", "?")
    print(f"\n=== JANGTQ_K 53K test ===")
    print(f"  wall:            {wall:.1f}s ({wall/60:.1f} min)")
    print(f"  prompt_tokens:   {ptoks}")
    print(f"  completion_tok:  {ctoks}")
    if isinstance(ctoks, (int, float)) and ctoks > 0:
        print(f"  gen t/s:         {ctoks/wall:.2f}")
    print(f"\n  reasoning_content[:400]: {rc[:400]!r}")
    print(f"\n  content[:400]:          {content[:400]!r}")
    combined = rc + " " + content
    bf = "BLUE-FALCON-48217" in combined
    print(f"\n  BLUE-FALCON-48217 recovered: {bf}")
    print(f"  warnings: {d.get('warnings', [])}")
    out = {"wall": wall, "usage": usage, "reasoning": rc, "content": content, "blue_falcon": bf}
    json.dump(out, open("/tmp/jangtq_53k_result.json", "w"), indent=2)
    print("# wrote /tmp/jangtq_53k_result.json")
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code} after {time.time()-t0:.1f}s: {e.read()[:400]}")
except Exception as e:
    print(f"ERR after {time.time()-t0:.1f}s: {e}")
