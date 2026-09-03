#!/usr/bin/env python3
"""Send real images to the dsvision engine and print what it says."""
import base64, json, mimetypes, os, sys, time, urllib.request
base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30021"
model = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4-flash-vision-exp"
D = os.path.expanduser("~/llm/dsvision/testimg")
TESTS = [
    ("shapes.png", "What exactly do you see in this image? List every shape with its colour, and read any text verbatim."),
    ("cat.jpg", "Describe this photo in two sentences. What animal is it and what is it doing?"),
    ("coco.png", "Describe this photo in two sentences. How many people and objects can you identify?"),
]
for fn, q in TESTS:
    p = os.path.join(D, fn)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        print(f"--- {fn}: missing, skipped"); continue
    mt = mimetypes.guess_type(p)[0] or "image/png"
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    body = json.dumps({"model": model, "max_tokens": 300, "temperature": 0.2,
        "chat_template_kwargs": {"thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}},
            {"type": "text", "text": q}]}]}).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            j = json.load(r)
        c = j["choices"][0]
        print(f"--- {fn}  ({time.time()-t0:.1f}s, {j['usage']['prompt_tokens']} prompt tok, finish={c['finish_reason']})")
        print((c["message"]["content"] or "(EMPTY)").strip()[:900])
    except urllib.error.HTTPError as e:
        print(f"--- {fn}: HTTP {e.code}: {e.read().decode()[:400]}")
    except Exception as e:
        print(f"--- {fn}: {type(e).__name__}: {e}")
    print()
