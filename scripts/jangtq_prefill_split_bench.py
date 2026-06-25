#!/usr/bin/env python3
"""
Split-measurement + prefill_batch_size sweep for JANGTQ_K (vMLX) on the 53K
BLUE-FALCON prompt. Backs JANGTQ_K_SPEED_PLAN.md US-A + US-C.

Measures what yesterday's single 23.8min wall never broke out:
  - Run 1 COLD: full prefill + decode (prefix cache empty)
  - Run 2 WARM: same prefix, different final question -> decode-only (cache hit)

Then, with a different VMLX_INFERENCE_PREFILL_BATCH_SIZE (set at server launch),
repeats the cold measurement and compares prefill tok/s.

Sends to /v1/chat/completions. Requires `model` field (vMLX 422s otherwise).
vMLX has no timings dict; derives tok/s from usage/wall.
"""
import json, time, urllib.request, urllib.error, sys, os

URL = "http://127.0.0.1:8082/v1/chat/completions"
MODEL = os.environ.get("VMLX_MODEL", "glm52-jangtq-k")
SRC = os.environ.get("LONGCTX_PROMPT", "/tmp/longctx_prompt.txt")
PBS = os.environ.get("VMLX_INFERENCE_PREFILL_BATCH_SIZE", "?")
THINK = os.environ.get("THINKING", "on") == "on"

prefix = open(SRC, encoding="utf-8").read()
print(f"# prompt: {SRC} ({len(prefix)} chars)")
print(f"# prefill_batch_size (env at launch): {PBS}  thinking: {THINK}")

def chat(question, max_tokens=400):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prefix + "\n\nQuestion: " + question}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    if not THINK:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    data = json.dumps(body).encode()
    t0 = time.time()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5400) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "body": e.read()[:300].decode(errors="replace")}
    wall = time.time() - t0
    d = json.loads(raw)
    msg = d["choices"][0]["message"]
    rc = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    usage = d.get("usage", {})
    pt = usage.get("prompt_tokens", "?")
    ct = usage.get("completion_tokens", "?")
    bf = "BLUE-FALCON-48217" in (rc + " " + content)
    return {
        "wall": wall, "prompt_tokens": pt, "completion_tokens": ct,
        "reasoning": rc[:200], "content": content[:200], "blue_falcon": bf,
        "warnings": d.get("warnings", []),
    }

def show(tag, r):
    if "error" in r:
        print(f"\n=== {tag} ===\n  ERR: {r['error']}  body: {r['body']}")
        return
    ct = r["completion_tokens"]
    dec = (ct / r["wall"]) if isinstance(ct, (int, float)) and ct and r["wall"] else None
    print(f"\n=== {tag} ===")
    print(f"  wall: {r['wall']:.1f}s   prompt_tok: {r['prompt_tokens']}   completion_tok: {r['completion_tokens']}")
    if dec is not None:
        print(f"  derived_total_tok/s (completion/wall): {dec:.2f}   [NOTE: includes prefill if cold; decode-only if warm]")
    print(f"  BLUE-FALCON recovered: {r['blue_falcon']}")
    print(f"  content[:200]: {r['content'][:200]!r}")

print("\n##### US-A: prefill/decode split (cold then warm, same session) #####")
r1 = chat("What is the hidden passphrase mentioned above? Answer in one short sentence.", max_tokens=400)
show("Run 1 COLD (prefill + decode)", r1)

r2 = chat("How many sections does the document above have? Answer in one short sentence.", max_tokens=400)
show("Run 2 WARM (decode-only, prefix cache hit)", r2)

if "wall" in r1 and "wall" in r2 and "completion_tokens" in r1 and isinstance(r1.get("completion_tokens"), (int, float)):
    cold_decode = r1["completion_tokens"] / r2["wall"] if r2.get("completion_tokens") else None
    pt = r1["prompt_tokens"] if isinstance(r1.get("prompt_tokens"), (int, float)) else None
    print(f"\n## US-A split (PBS={PBS})")
    if pt:
        # prefill_wall ≈ cold_wall - decode_wall (assume decode at same rate)
        if r2.get("completion_tokens") and r2.get("wall"):
            dec_rate = r2["completion_tokens"] / r2["wall"]
            est_decode_cold = r1["completion_tokens"] / dec_rate if dec_rate else 0
            est_prefill = r1["wall"] - est_decode_cold
            if est_prefill > 0:
                print(f"  est_pre_fill_wall: {est_prefill:.1f}s   est_pre_fill_tok/s: {pt/est_prefill:.2f}")
                print(f"  est_decode_wall (cold): {est_decode_cold:.1f}s   decode_tok/s: {dec_rate:.2f}")
            else:
                print(f"  decode_tok/s (warm): {dec_rate:.2f}")
    print(f"  warm_wall (decode-only {r2.get('completion_tokens')} tok): {r2.get('wall')}s")

out = {
    "prefill_batch_size": PBS, "thinking": THINK,
    "cold": r1, "warm": r2,
}
json.dump(out, open("/tmp/jangtq_split.json", "w"), indent=2)
print("# wrote /tmp/jangtq_split.json")

if __name__ == "__main__":
    pass
