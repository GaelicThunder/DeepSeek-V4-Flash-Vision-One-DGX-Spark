#!/usr/bin/env python3
"""2-bit vs 3-bit quality bench. Identical items + identical grading on every engine.

  bench_all.py OUT.json [--base http://127.0.0.1:30021] [--only mc,math,needle] [--conc 4]

mc     : MMLU-Pro 251 items, 10-way, scored by the logprob of the answer letter
         (one forward pass, no generation -> immune to token-budget artifacts).
         Run twice: options in original order and rotated by 3, so position bias
         and self-agreement are measured too.
math   : MATH-500 level 5, 60 items, free generation, \\boxed{} exact match.
needle : retrieval of a planted fact at 3 context lengths x 3 depths.
"""
import json, math, os, re, sys, time, urllib.request, datetime, threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
out_path = sys.argv[1]
def arg(k, d):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
BASE = arg("--base", "http://127.0.0.1:30021")
ONLY = set(arg("--only", "mc,math,needle").split(","))
CONC = int(arg("--conc", "4"))
LETTERS = "ABCDEFGHIJ"

def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

with urllib.request.urlopen(f"{BASE}/v1/models", timeout=60) as r:
    _m = json.load(r)["data"][0]
MODEL, MAXLEN = _m["id"], _m.get("max_model_len", 131072)
print(f"engine: {MODEL}  max_model_len={MAXLEN}  base={BASE}", flush=True)

RES = {"base": BASE, "model": MODEL, "max_model_len": MAXLEN,
       "when": datetime.datetime.now().isoformat(timespec="seconds")}
lock = threading.Lock()
done = [0]

def progress(tag, n):
    with lock:
        done[0] += 1
        if done[0] % 20 == 0 or done[0] == n:
            print(f"  {tag} {done[0]}/{n}", flush=True)

# ---------------------------------------------------------------- MC
MC_HEAD = ("The following is a multiple choice question. "
           "Reply with the single letter of the correct option.\n\n")

def mc_prompt(it, shift):
    n = len(it["opts"])
    order = [(i + shift) % n for i in range(n)]
    lines = [f"{LETTERS[j]}. {it['opts'][o]}" for j, o in enumerate(order)]
    gold_pos = order.index(it["gold"])
    p = MC_HEAD + f"Question: {it['q']}\n" + "\n".join(lines) + "\nAnswer:"
    return p, LETTERS[gold_pos], n

def mc_one(it, shift, n_total):
    p, gold, n = mc_prompt(it, shift)
    try:
        j = post("/v1/completions", {"model": MODEL, "prompt": p, "max_tokens": 1,
                                     "temperature": 0, "logprobs": 20})
        tl = (j["choices"][0]["logprobs"]["top_logprobs"] or [{}])[0]
        best, bestlp = None, -1e30
        for tok, lp in tl.items():
            s = tok.strip()
            if len(s) == 1 and s.upper() in LETTERS[:n] and lp > bestlp:
                best, bestlp = s.upper(), lp
        pred = best or "?"
        margin = None
        if best:
            others = [lp for tok, lp in tl.items()
                      if tok.strip().upper() != best and len(tok.strip()) == 1
                      and tok.strip().upper() in LETTERS[:n]]
            if others:
                margin = round(bestlp - max(others), 4)
        r = {"pred": pred, "gold": gold, "ok": pred == gold, "margin": margin,
             "ptok": j["usage"]["prompt_tokens"]}
    except Exception as e:
        r = {"pred": "!", "gold": gold, "ok": False, "err": f"{type(e).__name__}: {e}"}
    progress("mc", n_total)
    return r

if "mc" in ONLY:
    items = json.load(open(f"{HERE}/items_mc.json"))
    RES["mc"] = {}
    for shift in (0, 3):
        done[0] = 0
        t0 = time.time()
        with ThreadPoolExecutor(CONC) as ex:
            rows = list(ex.map(lambda it: mc_one(it, shift, len(items)), items))
        acc = sum(r["ok"] for r in rows)
        bycat = {}
        for it, r in zip(items, rows):
            b = bycat.setdefault(it["cat"], [0, 0]); b[1] += 1; b[0] += int(r["ok"])
        margins = [r["margin"] for r in rows if r.get("margin") is not None]
        RES["mc"][f"shift{shift}"] = {
            "n": len(items), "correct": acc, "pct": round(100 * acc / len(items), 2),
            "mean_margin": round(sum(margins) / len(margins), 4) if margins else None,
            "sec": round(time.time() - t0, 1),
            "by_cat": {k: f"{v[0]}/{v[1]}" for k, v in sorted(bycat.items())},
            "preds": [r["pred"] for r in rows], "golds": [r["gold"] for r in rows],
            "oks": [r["ok"] for r in rows]}
        print(f"MC shift{shift}: {acc}/{len(items)} = {100*acc/len(items):.2f}%  "
              f"margin={RES['mc'][f'shift{shift}']['mean_margin']}  "
              f"{RES['mc'][f'shift{shift}']['sec']}s", flush=True)
    a, b = RES["mc"]["shift0"]["oks"], RES["mc"]["shift3"]["oks"]
    RES["mc"]["both_correct"] = sum(x and y for x, y in zip(a, b))
    RES["mc"]["agreement"] = round(100 * sum(x == y for x, y in zip(a, b)) / len(a), 2)
    RES["mc"]["mean_pct"] = round((RES["mc"]["shift0"]["pct"] + RES["mc"]["shift3"]["pct"]) / 2, 2)
    print(f"MC mean={RES['mc']['mean_pct']}%  robust(both orders)={RES['mc']['both_correct']}/{len(a)}"
          f"  self-agreement={RES['mc']['agreement']}%", flush=True)
    json.dump(RES, open(out_path, "w"), indent=1)

# ---------------------------------------------------------------- MATH
def norm_ans(s):
    s = str(s).strip()
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\$", "").replace("$", "").replace(" ", "").replace("\\ ", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
    s = s.rstrip(".").strip()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s.lower()

def boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        m = re.findall(r"ANSWER:\s*(.+?)\s*$", text, re.M | re.I)
        return m[-1].strip() if m else ""
    j = text.find("{", i)
    if j < 0:
        return text[i + 6:i + 40]
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{": depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return text[j + 1:]

MATH_SUF = ("\n\nSolve it. Put the final answer in \\boxed{} on the last line.")

def math_one(it, n_total):
    try:
        j = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 2400, "temperature": 0,
                 "chat_template_kwargs": {"thinking": False},
                 "messages": [{"role": "user", "content": it["q"] + MATH_SUF}]})
        c = j["choices"][0]
        txt = c["message"]["content"] or ""
        got = boxed(txt)
        ok = norm_ans(got) == norm_ans(it["gold"])
        r = {"ok": ok, "got": got[:120], "gold": it["gold"], "tok": j["usage"]["completion_tokens"],
             "finish": c["finish_reason"]}
    except Exception as e:
        r = {"ok": False, "got": f"!{type(e).__name__}", "gold": it["gold"], "tok": 0, "finish": "error"}
    progress("math", n_total)
    return r

if "math" in ONLY:
    items = json.load(open(f"{HERE}/items_math.json"))
    done[0] = 0; t0 = time.time()
    with ThreadPoolExecutor(CONC) as ex:
        rows = list(ex.map(lambda it: math_one(it, len(items)), items))
    acc = sum(r["ok"] for r in rows)
    trunc = sum(r["finish"] == "length" for r in rows)
    RES["math"] = {"n": len(items), "correct": acc, "pct": round(100 * acc / len(items), 2),
                   "truncated": trunc, "mean_tok": round(sum(r["tok"] for r in rows) / len(rows), 1),
                   "sec": round(time.time() - t0, 1),
                   "items": {it["id"]: r for it, r in zip(items, rows)}}
    print(f"MATH: {acc}/{len(items)} = {100*acc/len(items):.2f}%  truncated={trunc}  "
          f"mean_tok={RES['math']['mean_tok']}  {RES['math']['sec']}s", flush=True)
    json.dump(RES, open(out_path, "w"), indent=1)

# ---------------------------------------------------------------- NEEDLE
FILLER = ("The regional archive stores maintenance reports for the northern rail corridor. "
          "Report {i} records routine track inspection, ballast condition and signal timing checks. "
          "No anomalies were escalated to the district engineer in that period. ")
CODES = {0.1: "QUARTZ-4417", 0.5: "MERIDIAN-8823", 0.9: "OBSIDIAN-1256"}

def build_ctx(target_tok, depth):
    approx_tok_per_block = 45
    nblocks = max(4, int(target_tok / approx_tok_per_block))
    blocks = [FILLER.format(i=i) for i in range(nblocks)]
    needle = (f"IMPORTANT: the access code for the {['north','central','south'][int(depth*10)%3]} "
              f"vault is {CODES[depth]}. Remember it. ")
    pos = min(nblocks - 1, max(0, int(nblocks * depth)))
    blocks.insert(pos, needle)
    return "".join(blocks)

if "needle" in ONLY:
    lens = [n for n in (4096, 32768, 131072) if n < MAXLEN - 8192]
    RES["needle"] = {}
    for L in lens:
        for depth in (0.1, 0.5, 0.9):
            ctx = build_ctx(L, depth)
            q = (ctx + "\n\nQuestion: what is the access code mentioned in the text above? "
                 "Reply with the code only.")
            t0 = time.time()
            try:
                j = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 24, "temperature": 0,
                         "chat_template_kwargs": {"thinking": False},
                         "messages": [{"role": "user", "content": q}]}, timeout=2400)
                txt = (j["choices"][0]["message"]["content"] or "").strip()
                ok = CODES[depth].lower() in txt.lower()
                pt = j["usage"]["prompt_tokens"]
            except Exception as e:
                txt, ok, pt = f"!{type(e).__name__}: {e}"[:80], False, 0
            RES["needle"][f"{L}@{depth}"] = {"ok": ok, "got": txt[:60], "prompt_tok": pt,
                                             "sec": round(time.time() - t0, 1)}
            print(f"needle {L:7d} depth {depth}: {'PASS' if ok else 'FAIL'} "
                  f"({pt} tok, {time.time()-t0:.0f}s) got={txt[:40]!r}", flush=True)
    n = len(RES["needle"]); k = sum(v["ok"] for v in RES["needle"].values())
    RES["needle_score"] = f"{k}/{n}"
    print(f"NEEDLE: {k}/{n}", flush=True)

json.dump(RES, open(out_path, "w"), indent=1)
print("wrote", out_path)
