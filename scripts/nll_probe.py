#!/usr/bin/env python3
"""Score a corpus token by token through `prompt_logprobs`, and compare two engines *paired*.

    scripts/nll_probe.py CORPUS.jsonl OUT.json [--base http://127.0.0.1:30021] [--model NAME]
    scripts/nll_probe.py --compare A.json B.json          # paired delta with its standard error

Every chunk is sent as a completion prompt with `max_tokens: 1` and `prompt_logprobs: 0`, so the
engine returns the log-probability of each prompt token under the model (the prefill path; no
sampling is involved). The per-token values are stored, which is what makes the comparison paired:
two engines that share a tokenizer score exactly the same tokens, so the noise that cancels is the
text's own difficulty and what remains is the difference between the two models.

With ~60k tokens the paired standard error of the mean NLL difference is ~0.0004-0.001 nats, which
resolves the per-layer quantization deltas this project argues about (~0.002-0.005 nats). The
8-passage `ppl_probe.py` cannot; keep it for quick sanity checks only.

`--compare` reports mean delta B-A (negative = B is better), the token-level paired SE, and a
chunk-level SE (chunks as the unit; conservative, robust to within-text correlation).
"""
import argparse
import datetime
import json
import math
import os
import sys
import time
import urllib.request


def post(base, path, body, timeout=900):
    req = urllib.request.Request(base.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def served_model(base):
    with urllib.request.urlopen(base.rstrip("/") + "/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def chunk_logprobs(base, model, text):
    res = post(base, "/v1/completions", {"model": model, "prompt": text, "max_tokens": 1,
                                          "temperature": 0, "prompt_logprobs": 0, "logprobs": 0})
    pl = res["choices"][0].get("prompt_logprobs")
    if pl is None:
        raise SystemExit("server returned no prompt_logprobs: " + json.dumps(res)[:400])
    lps = []
    for entry in pl[1:]:  # position 0 (BOS / first token) has no conditional logprob
        if not entry:
            continue
        vals = list(entry.values())
        lps.append(vals[0]["logprob"] if len(vals) == 1 else max(v["logprob"] for v in vals))
    return lps


def score(args):
    base = args.base
    model = args.model or served_model(base)
    rows = [json.loads(l) for l in open(os.path.expanduser(args.corpus)) if l.strip()]
    out = {"base": base, "model": model, "corpus": os.path.basename(args.corpus),
           "when": datetime.datetime.now().isoformat(timespec="seconds"), "chunks": []}
    tot, n_tot, t_start = 0.0, 0, time.time()
    for i, r in enumerate(rows):
        t0 = time.time()
        lps = chunk_logprobs(base, model, r["text"])
        nll = -sum(lps) / len(lps)
        out["chunks"].append({"id": r["id"], "source": r["source"], "n": len(lps), "nll": round(nll, 5),
                              "logprobs": [round(x, 4) for x in lps]})
        tot += -sum(lps); n_tot += len(lps)
        print(f"[{i + 1:3d}/{len(rows)}] {r['id']:14s} n={len(lps):5d} nll={nll:.4f}  {time.time() - t0:5.1f}s", flush=True)
    by_src = {}
    for c in out["chunks"]:
        s = by_src.setdefault(c["source"], [0.0, 0])
        s[0] += -sum(c["logprobs"]); s[1] += c["n"]
    out["per_source"] = {k: {"nll": round(v[0] / v[1], 5), "ppl": round(math.exp(v[0] / v[1]), 3), "tokens": v[1]} for k, v in by_src.items()}
    out["mean_nll"] = round(tot / n_tot, 5); out["mean_ppl"] = round(math.exp(tot / n_tot), 3); out["tokens"] = n_tot
    out["sec"] = round(time.time() - t_start, 1)
    json.dump(out, open(os.path.expanduser(args.out), "w"))
    print(f"\n{model}: mean NLL {out['mean_nll']:.5f}  ppl {out['mean_ppl']:.3f}  tokens {n_tot}  "
          f"per source: " + "  ".join(f"{k}={v['nll']:.4f}" for k, v in out["per_source"].items()))
    print(f"wrote {args.out}")


def compare(path_a, path_b):
    A = json.load(open(os.path.expanduser(path_a))); B = json.load(open(os.path.expanduser(path_b)))
    ca = {c["id"]: c for c in A["chunks"]}; cb = {c["id"]: c for c in B["chunks"]}
    ids = [i for i in ca if i in cb]
    if not ids:
        raise SystemExit("no common chunk ids")
    d_tok, d_chunk, by_src = [], [], {}
    for i in ids:
        la, lb = ca[i]["logprobs"], cb[i]["logprobs"]
        if len(la) != len(lb):
            raise SystemExit(f"chunk {i}: token counts differ ({len(la)} vs {len(lb)}) -- different tokenizers?")
        d = [(-b) - (-a) for a, b in zip(la, lb)]  # delta NLL per token, B - A
        d_tok.extend(d)
        d_chunk.append(sum(d) / len(d))
        s = by_src.setdefault(ca[i]["source"], [])
        s.extend(d)
    n = len(d_tok)
    mean = sum(d_tok) / n
    var = sum((x - mean) ** 2 for x in d_tok) / (n - 1)
    se_tok = math.sqrt(var / n)
    m_c = sum(d_chunk) / len(d_chunk)
    se_chunk = math.sqrt(sum((x - m_c) ** 2 for x in d_chunk) / (len(d_chunk) - 1) / len(d_chunk)) if len(d_chunk) > 1 else float("nan")
    print(f"A = {A['model']} ({A.get('base')})  mean NLL {A['mean_nll']:.5f}")
    print(f"B = {B['model']} ({B.get('base')})  mean NLL {B['mean_nll']:.5f}")
    print(f"delta NLL (B - A) over {n} paired tokens: {mean:+.5f} nats  "
          f"token-SE {se_tok:.5f} (z={mean / se_tok:+.1f})  chunk-SE {se_chunk:.5f} (z={mean / se_chunk:+.1f}, {len(d_chunk)} chunks)")
    for src, d in by_src.items():
        m = sum(d) / len(d); v = sum((x - m) ** 2 for x in d) / (len(d) - 1)
        print(f"  {src:9s} {m:+.5f}  SE {math.sqrt(v / len(d)):.5f}  n={len(d)}")
    print("negative = B better")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--base", default="http://127.0.0.1:30021")
    ap.add_argument("--model", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if not args.corpus or not args.out:
        ap.error("CORPUS.jsonl and OUT.json are required unless --compare is used")
    score(args)


if __name__ == "__main__":
    main()
