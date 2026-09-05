#!/usr/bin/env python3
"""Build a fixed, public, >=50k-token corpus for paired NLL comparisons between packs.

Why: the 8-passage `ppl_probe.py` has ~1k tokens. The per-layer quantization deltas this project
argues about (3->2 bit on one layer, 40 experts restored at 2 bit) are ~0.002-0.005 nats, and at 1k
tokens the paired standard error is ~7x larger than at 50k. This corpus makes those deltas
measurable: score it with `scripts/nll_probe.py` on two engines and compare token by token.

Sources (all public, ungated on the Hugging Face Hub, fixed split and order, no shuffling):
  wikitext  Salesforce/wikitext  wikitext-103-raw-v1  test   English prose (~60 % of the budget)
  gsm8k     openai/gsm8k         main                 test   math word problems + solutions (~15 %)
  code      google/code_x_glue_ct_code_to_text  python  test  Python functions (~25 %)

Usage:
  scripts/nll_corpus.py OUT.jsonl --tokenizer ~/models/<pack>/tp1 [--target-tokens 60000] [--chunk 3072]

Each output row: {"id", "source", "n_tokens", "text"}. Chunks are cut on paragraph / item boundaries
and never exceed --chunk tokens, so every chunk prefills in one pass. Token counts use the served
model's own tokenizer, so two engines that share the tokenizer score exactly the same token sequence.
"""
import argparse
import json
import os
import sys


def load_texts(source, limit_chars):
    from datasets import load_dataset  # lazy: only needed when building

    if source == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        # wikitext is one line per paragraph with blank separators and " = Title = " headers;
        # rebuild articles as paragraphs, drop headers and empty lines
        buf, out = [], []
        for line in ds["text"]:
            s = line.strip()
            if not s:
                continue
            if s.startswith("=") and s.endswith("="):
                if buf:
                    out.append("\n\n".join(buf))
                    buf = []
                continue
            buf.append(s)
        if buf:
            out.append("\n\n".join(buf))
        texts = out
    elif source == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(ds["question"], ds["answer"])]
    elif source == "code":
        ds = load_dataset("google/code_x_glue_ct_code_to_text", "python", split="test")
        texts = [c for c in ds["original_string"] if 200 <= len(c) <= 6000]
    else:
        raise SystemExit(f"unknown source {source}")
    acc, total = [], 0
    for t in texts:
        acc.append(t)
        total += len(t)
        if total >= limit_chars:
            break
    return acc


def chunk_texts(tok, texts, chunk_tokens, budget_tokens, source, start_id):
    """Pack consecutive items into chunks of <= chunk_tokens; never split an item unless it alone
    exceeds the chunk size (then cut it on token boundaries)."""
    rows, cur, cur_n, used = [], [], 0, 0
    sep = "\n\n"
    sep_n = len(tok.encode(sep, add_special_tokens=False))

    def flush():
        nonlocal cur, cur_n, used
        if not cur:
            return
        text = sep.join(cur)
        n = len(tok.encode(text, add_special_tokens=False))
        rows.append({"id": f"{source}-{start_id + len(rows):03d}", "source": source, "n_tokens": n, "text": text})
        used += n
        cur, cur_n = [], 0

    for t in texts:
        if used >= budget_tokens:
            break
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) > chunk_tokens:
            flush()
            for i in range(0, len(ids), chunk_tokens):
                if used >= budget_tokens:
                    break
                piece = tok.decode(ids[i:i + chunk_tokens])
                n = len(tok.encode(piece, add_special_tokens=False))
                rows.append({"id": f"{source}-{start_id + len(rows):03d}", "source": source, "n_tokens": n, "text": piece})
                used += n
            continue
        if cur and cur_n + sep_n + len(ids) > chunk_tokens:
            flush()
        cur.append(t)
        cur_n += len(ids) + (sep_n if len(cur) > 1 else 0)
    if used < budget_tokens:
        flush()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--tokenizer", required=True, help="directory with tokenizer.json of the served model")
    ap.add_argument("--target-tokens", type=int, default=60000)
    ap.add_argument("--chunk", type=int, default=3072, help="max tokens per chunk (one prefill pass)")
    ap.add_argument("--mix", default="wikitext:0.60,gsm8k:0.15,code:0.25")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.expanduser(args.tokenizer), trust_remote_code=True)
    mix = [(s.split(":")[0], float(s.split(":")[1])) for s in args.mix.split(",")]
    rows = []
    for source, share in mix:
        budget = int(args.target_tokens * share)
        texts = load_texts(source, limit_chars=budget * 6)  # ~4 chars/token, generous
        part = chunk_texts(tok, texts, args.chunk, budget, source, 0)
        got = sum(r["n_tokens"] for r in part)
        print(f"{source:9s} {len(part):3d} chunks  {got:6d} tokens  (budget {budget})", flush=True)
        rows.extend(part)
    with open(os.path.expanduser(args.out), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = sum(r["n_tokens"] for r in rows)
    print(f"wrote {len(rows)} chunks, {total} tokens -> {args.out}")


if __name__ == "__main__":
    main()
