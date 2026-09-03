#!/usr/bin/env python3
"""Decode-speed probe (single stream) via /v1/chat/completions: code@T0 and prose@T1, 300 tokens each."""
import json, sys, time, urllib.request, datetime
base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30021"
model = sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else None
runs = int(sys.argv[4]) if len(sys.argv) > 4 else 2
PROMPTS = {
 "code_T0": (0.0, "Write a Python module implementing an LRU cache class with get/put, O(1) operations, type hints and a small test. Only code, no explanation."),
 "prose_T1": (1.0, "Scrivi un racconto breve (circa 400 parole) ambientato in un porto italiano all'alba, in prima persona, con un finale inaspettato."),
}
res = {"base": base, "model": model, "when": datetime.datetime.now().isoformat(timespec="seconds"), "runs": {}}
for name, (temp, prompt) in PROMPTS.items():
    rows = []
    for r in range(runs):
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 300,
                           "temperature": temp, "chat_template_kwargs": {"thinking": False}}).encode()
        req = urllib.request.Request(f"{base}/v1/chat/completions", body, {"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as rr:
            j = json.load(rr)
        dt = time.time() - t0
        n = j["usage"]["completion_tokens"]; p = j["usage"]["prompt_tokens"]
        rows.append({"tok": n, "prompt_tok": p, "sec": round(dt, 2), "tps": round(n / dt, 1)})
        print(f"{name} run{r}: {n} tok in {dt:.1f}s -> {n/dt:.1f} tok/s (incl. prefill of {p} tok)", flush=True)
    res["runs"][name] = rows
if out:
    json.dump(res, open(out, "w"), indent=1)
