"""Greedy chat sanity check against a running vLLM OpenAI server (stdlib only; used for the real-weight runs).

Sends a few temperature-0 chat requests (short factual questions, a haiku, and one ~6k-token document question whose
prefill chunks take the large all-reduce path), writes the raw replies to --out and prints a coherence verdict per prompt:
non-empty, mostly printable text, not a degenerate repetition loop, and (where the answer is known) the expected keyword.

usage: python3 chat_check.py --port 8100 --model <repo> --out chat.json
"""
import argparse
import json
import re
import sys
import time
import urllib.request

PASSAGE = (
    "The history of the lighthouse is a history of human attempts to make the sea less dangerous. The earliest beacons "
    "were simple fires lit on hilltops by fishermen who wanted their neighbours to find the harbour after dark. As trade "
    "grew, the fires moved to towers, and the towers grew taller, until the Pharos of Alexandria rose above the "
    "Mediterranean and became one of the wonders of the ancient world. The invention of the Argand oil lamp in the late "
    "eighteenth century made the flame brighter and steadier, but it was Augustin Fresnel who transformed the lighthouse "
    "in 1822 with a lens built from concentric rings of glass prisms. Today most lighthouses are automatic, powered by "
    "solar panels and monitored from distant offices, and satellite navigation has taken over much of their work."
)

PROMPTS = [
    ("capital", "What is the capital of France? Answer in one short sentence.", ["paris"]),
    ("arith", "What is 17 times 23? Reply with the number and one sentence of explanation.", ["391"]),
    ("sky", "Explain in two sentences why the sky is blue.", ["scatter"]),
    ("haiku", "Write a haiku about the sea.", []),
    ("long", "Here is a text, repeated several times.\n\n" + "\n\n".join([PASSAGE] * 28) +
     "\n\nIn one sentence, what is this text about?", ["lighthouse"]),
]


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def coherent(text, keywords):
    t = text.strip()
    if len(t) < 5:
        return False, "empty or too short"
    printable = sum(1 for c in t if c.isprintable() or c in "\n\t") / len(t)
    if printable < 0.98:
        return False, f"printable fraction {printable:.2f}"
    words = re.findall(r"\w+", t.lower())
    if len(words) >= 20 and len(set(words)) / len(words) < 0.2:
        return False, f"repetition loop (unique word ratio {len(set(words)) / len(words):.2f})"
    low = t.lower()
    miss = [k for k in keywords if k not in low]
    if miss:
        return False, f"missing keyword {miss}"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    rows, good = [], 0
    for name, prompt, kw in PROMPTS:
        t0 = time.time()
        try:
            r = post(url, {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                           "max_tokens": args.max_tokens, "seed": 1}, 1800)
        except Exception as e:  # keep going: report every prompt
            rows.append({"name": name, "error": repr(e)})
            print(f"CHAT {name}: ERROR {e!r}", flush=True)
            continue
        msg = r["choices"][0]["message"]
        text = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or msg.get("reasoning") or "")
        ok, why = coherent(text, kw)
        good += ok
        usage = r.get("usage", {})
        rows.append({"name": name, "ok": ok, "why": why, "prompt_tokens": usage.get("prompt_tokens"),
                     "completion_tokens": usage.get("completion_tokens"), "seconds": round(time.time() - t0, 2),
                     "reply": msg})
        print(f"CHAT {name}: {'OK' if ok else 'FAIL'} ({why}) prompt_tokens={usage.get('prompt_tokens')} "
              f"{time.time() - t0:.1f}s :: {text.strip()[:220]!r}", flush=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"CHAT_SUMMARY ok={good}/{len(PROMPTS)}", flush=True)
    return 0 if good == len(PROMPTS) else 1


if __name__ == "__main__":
    sys.exit(main())
