#!/usr/bin/env python3
"""Paired divergence between the decode path and the prefill path of one engine, on the model's own tokens.

    tools/decode_vs_prefill.py --model <served-name> [--base URL] [-n 400] [--top 20] [--norms head_norms.npy] [--out X.json]

For each prompt: generate n tokens greedily with top-K logprobs (these rows come from the DECODE path --
the speculative verify block when speculation is on, single-token decode otherwise), then re-score the
identical prompt+generation with `prompt_logprobs` (the PREFILL path, one pass). For every generated token
we then hold two log-probabilities of the same token under the same prefix and the same weights, so

    delta_i = logprob_decode(token_i) - logprob_prefill(token_i)

is zero up to kernel numerics if the two paths agree. `token_leak_probe.py` flags the extreme cases; this
tool reports the whole distribution: mean, standard error, tail counts, the worst positions, and -- when a
row-norm file for `head.weight` is given -- whether |delta| over the shared top-K tokens grows with the
head row norm (a diffuse, precision-type fault in the logits GEMV) or sits on a few tokens (an upstream,
direction-type fault). Run it once per kernel configuration and compare.

Cost: n tokens of generation + one prefill per prompt (~2 min for 6 x 400 tokens at 25 tok/s).
"""
import argparse
import json
import math
import os
import statistics as st
import sys
import urllib.error
import urllib.request

PROMPTS = [
    "Write a 250-word reflective blog post about maintaining a small home server, with several short sentences and lists. Mention electricity cost.",
    "Explain to a curious teenager why the sky is blue and sunsets are red, in about 300 words, no bullet points.",
    "Write a Python module implementing an LRU cache with per-entry TTL, with docstrings and a small pytest test. Output only code.",
    "Summarize the plot of a heist film that does not exist yet: give it a title, three characters and a twist, in 250 words.",
    "Describe, step by step, how to debug a Linux service that fails to start after a reboot. Use numbered steps and short sentences.",
    "Write a short dialogue between two colleagues arguing about whether to rewrite a legacy system, about 300 words.",
]


def post(base, path, body, timeout=900):
    req = urllib.request.Request(base.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit("HTTP %d on %s: %s" % (e.code, path, e.read()[:400].decode("utf8", "replace")))


def tid(s):
    """'token_id:123' -> 123 (return_tokens_as_token_ids), else None."""
    return int(s.split(":", 1)[1]) if isinstance(s, str) and s.startswith("token_id:") else None


def one_prompt(base, model, prompt, n, top, thinking):
    P = post(base, "/tokenize", {"model": model, "messages": [{"role": "user", "content": prompt}],
                                 "add_generation_prompt": True, "chat_template_kwargs": {"thinking": thinking}})["tokens"]
    g = post(base, "/v1/completions", {"model": model, "prompt": P, "max_tokens": n, "temperature": 0,
                                       "logprobs": top, "return_tokens_as_token_ids": True})["choices"][0]
    lp = g["logprobs"]
    gen_ids = [tid(t) for t in lp["tokens"]]
    if any(i is None for i in gen_ids):
        raise SystemExit("server did not honour return_tokens_as_token_ids; cannot pair tokens exactly")
    dec_lp = lp["token_logprobs"]
    dec_top = [{tid(k): v for k, v in (d or {}).items()} for d in lp["top_logprobs"]]
    r = post(base, "/v1/completions", {"model": model, "prompt": P + gen_ids, "max_tokens": 1, "temperature": 0,
                                       "prompt_logprobs": top, "logprobs": 0})["choices"][0]
    pl = r.get("prompt_logprobs")
    if pl is None:
        raise SystemExit("server returned no prompt_logprobs")
    rows = []
    for i, tok in enumerate(gen_ids):
        entry = pl[len(P) + i] or {}
        pre = {int(k): v["logprob"] for k, v in entry.items()}
        if tok not in pre:
            continue
        shared = {t: dec_top[i][t] - pre[t] for t in dec_top[i] if t in pre and t != tok}
        rows.append({"pos": i, "tok": tok, "text": entry[str(tok)].get("decoded_token", ""), "dec": dec_lp[i], "pre": pre[tok],
                     "delta": dec_lp[i] - pre[tok], "pre_rank": entry[str(tok)].get("rank"), "shared": shared})
    return rows, g.get("finish_reason")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:30021")
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", "--tokens", type=int, default=400)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS), help="how many of the built-in prompts to use")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--norms", help="npy file with the L2 norm of every head.weight row (see --help of token_leak_probe --head)")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out")
    a = ap.parse_args()
    norms = None
    if a.norms:
        import numpy as np
        norms = np.load(os.path.expanduser(a.norms))
    allrows = []
    for k, p in enumerate(PROMPTS[:a.prompts]):
        rows, fin = one_prompt(a.base, a.model, p, a.tokens, a.top, a.thinking)
        for r in rows:
            r["prompt"] = k
        allrows.extend(rows)
        print(f"  prompt {k}: {len(rows)} paired tokens (finish={fin})", flush=True)
    d = [r["delta"] for r in allrows]
    n = len(d)
    mean = sum(d) / n
    sd = st.pstdev(d)
    se = sd / math.sqrt(n)
    tails = {t: sum(1 for x in d if abs(x) > t) for t in (0.5, 1, 2, 5)}
    print(f"\n[{a.tag}] paired tokens n={n}  mean delta (decode-prefill) {mean:+.4f} nats  SD {sd:.4f}  SE {se:.4f}  "
          f"mean|delta| {sum(abs(x) for x in d) / n:.4f}")
    print(f"[{a.tag}] |delta| > 0.5: {tails[0.5]}  > 1: {tails[1]}  > 2: {tails[2]}  > 5: {tails[5]}   "
          f"(fraction > 1 nat: {100 * tails[1] / n:.2f} %)")
    worst = sorted(allrows, key=lambda r: -abs(r["delta"]))[:8]
    for r in worst:
        print(f"    p{r['prompt']} pos {r['pos']:3d} {r['text']!r:14s} decode p {math.exp(r['dec']):.3f}  prefill p {math.exp(r['pre']):.2e} rank {r['pre_rank']}  delta {r['delta']:+.2f}")
    # shared-top-K comparison: does |delta| grow with the head row norm?
    shared = [(t, dl) for r in allrows for t, dl in r["shared"].items()]
    if shared:
        sd_sh = st.pstdev([x for _, x in shared])
        print(f"[{a.tag}] shared top-{a.top} tokens n={len(shared)}  mean|delta| {sum(abs(x) for _, x in shared) / len(shared):.4f}  SD {sd_sh:.4f}")
        if norms is not None:
            import numpy as np
            nn = np.array([norms[t] for t, _ in shared]); dd = np.abs(np.array([x for _, x in shared]))
            q = np.quantile(nn, [0.2, 0.4, 0.6, 0.8])
            bins = np.digitize(nn, q)
            line = "  ".join(f"q{b + 1}(norm<={(q[b] if b < 4 else nn.max()):.1f}): {dd[bins == b].mean():.4f}" for b in range(5) if (bins == b).any())
            corr = float(np.corrcoef(nn, dd)[0, 1])
            print(f"[{a.tag}] mean|delta| by head-row-norm quintile: {line}   corr(|delta|, norm) = {corr:+.3f}")
    if a.out:
        json.dump({"tag": a.tag, "model": a.model, "n": n, "mean_delta": mean, "sd": sd, "se": se, "tails": tails,
                   "rows": [{k: v for k, v in r.items() if k != "shared"} for r in allrows]}, open(os.path.expanduser(a.out), "w"))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
