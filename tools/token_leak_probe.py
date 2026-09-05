#!/usr/bin/env python3
"""Find tokens the decode path emits that the prefill path would never choose.

Generate greedily, then re-score the *same* token sequence through the prefill path. At
temperature 0 the sampler can only take the argmax of the row it was handed, so a token the
model emitted with high confidence that prefill scores near zero means the two paths disagree
about that logits row -- a decode-path fault, not a sampling choice.

That is how the `)Skip` artifact of DeepSeek-V4-Flash was localized (decode p 0.65, prefill
p 0.007, deterministic on repeat). See docs/DECODE_PATH_TOKEN_LEAK.md. The probe knows no token
ids in advance, so it finds the same class of fault on any model and any OpenAI-compatible
engine that honours `prompt_logprobs`.

    # generate 400 greedy tokens and re-score them
    python3 tools/token_leak_probe.py --base http://127.0.0.1:30021 --model <served-name>

    # reproduce one known slot: seed the generation with the text that precedes it
    python3 tools/token_leak_probe.py --model <name> --prefix-text "$(cat prefix.txt)" -n 4 --verify-fix

    # judge a generation you already have, without generating anything
    python3 tools/token_leak_probe.py --model <name> --score-text output.txt

    # row-norm outliers of head.weight (a leaked token is usually one of the longest rows)
    python3 tools/token_leak_probe.py --model <name> --head /models/tp1 -n 0

A flagged token needs all three of: the model was confident (decode p >= --min-decode-p),
prefill disagrees (prefill p <= --max-prefill-p), and the gap is large (ratio >= --min-ratio).
Ordinary prefill/decode numerical drift on a flat distribution moves probabilities by about a
factor of two and is not reported. `--verify-fix` then replays each flagged slot through the
decode path with and without the mask, which separates a real leak from drift.

One sequential request at a time; safe to run against a serving box.
"""
import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request

DEFAULT_PROMPT = (
    "Write a 250-word reflective blog post about maintaining a small home server, "
    "with several short sentences and lists. Mention electricity cost."
)


def post(base, path, body, timeout=900):
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit("HTTP %d on %s: %s" % (e.code, path, e.read()[:500].decode("utf8", "replace")))
    except urllib.error.URLError as e:
        raise SystemExit("cannot reach %s: %s" % (base, e))


def prompt_ids(base, model, prompt, thinking):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True}
    if not thinking:
        body["chat_template_kwargs"] = {"thinking": False}
    return post(base, "/tokenize", body)["tokens"]


def tok(base, model, text):
    return post(base, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]


def detok(base, model, ids):
    return post(base, "/detokenize", {"model": model, "tokens": [int(i) for i in ids]})["prompt"]


def generate(base, model, ids, n):
    """Greedy continuation of raw token ids. Returns (generated ids, per-token decode p)."""
    r = post(base, "/v1/completions", {
        "model": model, "prompt": ids, "max_tokens": n, "temperature": 0, "logprobs": 1,
    })
    ch = r["choices"][0]
    gen = ch.get("token_ids")                      # exact ids when the engine reports them
    if not gen:
        gen = tok(base, model, ch["text"])         # otherwise re-tokenize the text
    lp = (ch.get("logprobs") or {}).get("token_logprobs") or []
    return list(gen), [math.exp(x) for x in lp if x is not None]


def prefill_scores(base, model, ids, first):
    """One prefill pass over the whole sequence. Per position >= first:
    (logprob of the actual token, its rank, [(text, p) ...] the top alternatives)."""
    r = post(base, "/v1/completions", {
        "model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 5,
    })
    # vLLM returns prompt_logprobs at the top level on /v1/chat/completions but inside the
    # choice on /v1/completions; forks differ, so accept either.
    pl = r.get("prompt_logprobs") or (r.get("choices") or [{}])[0].get("prompt_logprobs")
    if not pl:
        raise SystemExit("engine returned no prompt_logprobs "
                         "(needs a vLLM-compatible /v1/completions that honours prompt_logprobs)")
    if len(pl) != len(ids):
        raise SystemExit("prompt_logprobs has %d entries for %d tokens: tokenization drifted"
                         % (len(pl), len(ids)))
    out = []
    for pos in range(first, len(ids)):
        entry = pl[pos] or {}
        e = entry.get(str(ids[pos]))
        if e is None:
            out.append((None, None, []))
            continue
        alts = sorted(((v.get("logprob", -99.0), v.get("decoded_token")) for v in entry.values()),
                      reverse=True)[:3]
        out.append((e.get("logprob"), e.get("rank"), [(t, math.exp(l)) for l, t in alts]))
    return out


def head_outliers(tp1, top=12):
    """Row-norm outliers of head.weight, read straight from the safetensors shard."""
    try:
        import numpy as np
    except ImportError:
        raise SystemExit("--head needs numpy")
    import struct

    idx_path = os.path.join(tp1, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        wmap = json.load(open(idx_path))["weight_map"]
        name = next((k for k in wmap if k.endswith("head.weight") and "mtp" not in k), None)
        if name is None:
            raise SystemExit("no head.weight in %s" % idx_path)
        shard = os.path.join(tp1, wmap[name])
    else:
        shard, name = os.path.join(tp1, "model.safetensors"), "head.weight"
    with open(shard, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    meta = header.get(name)
    if meta is None:
        raise SystemExit("%s not found in %s" % (name, shard))
    if meta["dtype"] != "BF16":
        raise SystemExit("%s is %s, expected BF16" % (name, meta["dtype"]))
    s0 = meta["data_offsets"][0]
    rows, cols = meta["shape"]
    mm = np.memmap(shard, dtype=np.uint16, mode="r", offset=8 + hlen + s0, shape=(rows, cols))
    norms = np.empty(rows, np.float32)
    for i in range(0, rows, 4096):                 # chunked: never materialize the head
        w = (np.array(mm[i:i + 4096]).astype(np.uint32) << 16).view(np.float32)
        norms[i:i + 4096] = np.linalg.norm(w, axis=1)
    med = float(np.median(norms))
    print("\nhead.weight %s %s: median row norm %.3f, p99 %.3f, max %.3f"
          % (name, meta["shape"], med, float(np.percentile(norms, 99)), float(norms.max())))
    print("  longest rows -- a leaked token is usually one of these:")
    for r in np.argsort(-norms)[:top]:
        print("    id %-7d norm %6.3f  (%.2fx median)" % (int(r), norms[r], norms[r] / med))
    return norms


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:30021", help="engine base URL")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("-n", "--tokens", type=int, default=400, help="greedy tokens to generate (0 = none)")
    ap.add_argument("--prefix-text", metavar="TEXT",
                    help="treat this as already generated, then continue from it")
    ap.add_argument("--score-text", metavar="TEXT_OR_FILE",
                    help="score an existing generation and generate nothing")
    ap.add_argument("--min-decode-p", type=float, default=0.25,
                    help="the model must have been this confident (default 0.25)")
    ap.add_argument("--max-prefill-p", type=float, default=0.05,
                    help="prefill must disagree this strongly (default 0.05)")
    ap.add_argument("--min-ratio", type=float, default=20.0,
                    help="decode p / prefill p (default 20)")
    ap.add_argument("--thinking", action="store_true", help="leave the thinking block on")
    ap.add_argument("--head", metavar="TP1_DIR", help="also report head.weight row-norm outliers")
    ap.add_argument("--verify-fix", action="store_true",
                    help="replay each flagged slot through the decode path, with and without bad_words")
    a = ap.parse_args()

    P = prompt_ids(a.base, a.model, a.prompt, a.thinking)
    decode_p = []

    if a.score_text is not None:
        text = open(a.score_text).read() if os.path.exists(a.score_text) else a.score_text
        G = tok(a.base, a.model, text)
        print("scoring %d existing tokens through the prefill path ...\n" % len(G))
    else:
        G = tok(a.base, a.model, a.prefix_text) if a.prefix_text else []
        seeded = len(G)
        if a.tokens > 0:
            print("generating %d greedy tokens%s ..."
                  % (a.tokens, " after the seed prefix" if seeded else ""))
            gen, dp = generate(a.base, a.model, P + G, a.tokens)
            G = G + gen
            decode_p = [None] * seeded + dp
        if not G:
            if a.head:
                head_outliers(a.head)
                return 0
            raise SystemExit("nothing to score: pass -n > 0, --prefix-text or --score-text")
        print("re-scoring the generation through the prefill path ...\n")

    scores = prefill_scores(a.base, a.model, P + G, len(P))

    rows = []
    for i, (lp, rank, alts) in enumerate(scores):
        pp = math.exp(lp) if lp is not None else 0.0
        dp = decode_p[i] if i < len(decode_p) and decode_p[i] is not None else None
        if dp is None:                     # no decode probability: fall back to "prefill hates it"
            flag = pp <= a.max_prefill_p and (rank is None or rank > 5)
        else:
            flag = (dp >= a.min_decode_p and pp <= a.max_prefill_p
                    and pp > 0 and dp / pp >= a.min_ratio)
        if flag:
            rows.append((i, dp, pp, rank, alts))

    if not rows:
        print("no disagreement: every emitted token is one the prefill path also finds plausible.")
        if a.head:
            head_outliers(a.head)
        return 0

    print("  %3s  %-16s %9s %11s %6s %8s" %
          ("pos", "emitted", "decode p", "prefill p", "rank", "ratio"))
    for i, dp, pp, rank, alts in rows:
        emitted = detok(a.base, a.model, [G[i]])
        print("  %3d  %-16r %9s %11.2e %6s %8s   <== SUSPECT" %
              (i, emitted, ("%.3f" % dp) if dp is not None else "n/a", pp,
               rank if rank is not None else ">5",
               ("%.0fx" % (dp / pp)) if (dp and pp) else "-"))
        print("     context: ...%s" % detok(a.base, a.model, G[max(0, i - 10):i])[-64:].replace("\n", " "))
        if alts:
            print("     prefill wanted: %s" % ", ".join("%r %.3f" % (t, p) for t, p in alts))
    words = sorted({detok(a.base, a.model, [G[i]]) for i, _, _, _, _ in rows})
    print("\n%d suspect token%s in %d (%.2f %%). Mitigate with: \"bad_words\": %s"
          % (len(rows), "" if len(rows) == 1 else "s", len(G),
             100.0 * len(rows) / len(G), json.dumps(words)))

    if a.verify_fix:
        print("\nreplaying each slot through the DECODE path (prefix cut by one, two tokens):")
        for i, _, _, _, _ in rows:
            if i < 1:
                continue
            base_ids = P + G[:i - 1]
            out = {}
            for label, extra in (("stock", {}), ("masked", {"bad_words": words})):
                r = post(a.base, "/v1/completions", dict(
                    {"model": a.model, "prompt": base_ids, "max_tokens": 2, "temperature": 0}, **extra))
                out[label] = r["choices"][0]["text"]
            emitted = detok(a.base, a.model, [G[i]])
            if emitted not in out["stock"]:
                verdict = "not reproduced -- drift, not a leak"
            elif emitted in out["masked"]:
                verdict = "REPRODUCED and still there -- bad_words did not cover it"
            else:
                verdict = "REPRODUCED, and the mask removes it"
            print("  pos %-4d stock=%-30r masked=%-26r  %s" % (i, out["stock"], out["masked"], verdict))

    if a.head:
        head_outliers(a.head)
    return 1


if __name__ == "__main__":
    sys.exit(main())
