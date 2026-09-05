#!/usr/bin/env python3
"""
bench-ctx.py — prefill e decode MISURATI a contesto lungo, su qualunque endpoint
OpenAI-compatibile (llama.cpp :30001, vLLM/SparkInfer :8000, ds4 :8000).

Perche' non bastano i numeri del forum: il prefill dichiarato (1000 t/s) e il
nostro (151 t/s appuntato per dspec) non sono mai stati misurati con lo stesso
metodo. Qui il metodo e' uno solo, per tutti:

  - il prompt viene costruito con IL TOKENIZER DEL MODELLO fino a N token reali
    (nessuna stima "4 char per token": a 250k un errore del 10% sposta tutto);
  - ogni run ha un SALT unico in testa al prompt => il prefix-cache non puo'
    restituire un prefill falso (vLLM ha PREFIX_CACHE=1, llama.cpp ha lo slot
    reuse: senza salt la seconda misura e' aria);
  - stream=True: il primo chunk separa prefill da decode senza doverli dedurre.

    prefill t/s = token_prompt / (t_primo_chunk - t_invio)
    decode  t/s = (completion_tokens - 1) / (t_ultimo_chunk - t_primo_chunk)

  completion_tokens viene da `usage` (stream_options.include_usage). NON si contano i
  chunk: con lo speculative decoding un chunk porta piu' token (fino a k+1 per step) e
  il server puo' fondere piu' step in un chunk, quindi "chunk/s" NON e' "token/s".
  (Le prime misure del 2026-09-03 contavano i chunk: quei 10-11 "tok/s" erano step/s.)

  Il prompt chiede una risposta di ~200 parole e spegne il thinking, cosi' il decode
  misurato e' un tratto lungo e stabile (T=0) e non 6 token prima dell'EOS.

Il decode qui e' decode A CONTESTO PIENO, che e' l'unico numero onesto: i
"38 t/s" pubblicati sono C1 a contesto corto e non sono confrontabili con un
daily che lavora a 100k+.

Uso:
  ./bench-ctx.py --url http://127.0.0.1:8000 --model deepseek-v4-flash-0731-spark \
                 --tokenizer ~/code/sparkinfer/data/source --ctx 100000 250000
"""
import argparse
import json
import os
import sys
import time
import urllib.request

FILLER = (
    "The storage subsystem coordinator reconciles pending write intents against "
    "the durable manifest before acknowledging the commit, so a crash between "
    "the intent log and the manifest flush cannot surface a torn record to a "
    "reader that arrives afterwards. Each shard keeps its own monotonic epoch. "
)


def build_prompt(tok, target_tokens, salt):
    """Costruisce un prompt di ESATTAMENTE ~target_tokens token reali."""
    head = f"[run-salt {salt}] Read the following log and answer at the end.\n"
    n_head = len(tok.encode(head))
    n_filler = len(tok.encode(FILLER))
    tail = ("\n\nQuestion: in about 200 words of plain prose, what does the coordinator reconcile, "
            "why does the ordering of the intent log and the manifest flush matter, and what role "
            "does the per-shard epoch play? Do not use lists.")
    n_tail = len(tok.encode(tail))
    reps = max(1, (target_tokens - n_head - n_tail) // n_filler)
    body = FILLER * reps
    prompt = head + body + tail
    n = len(tok.encode(prompt))
    # aggiustamento fine: aggiunge/toglie ripetizioni finche' non siamo entro l'1%
    while n < target_tokens * 0.99:
        reps += max(1, int((target_tokens - n) / n_filler))
        prompt = head + FILLER * reps + tail
        n = len(tok.encode(prompt))
    return prompt, n


def run_one(url, model, prompt, n_prompt, max_tokens, timeout, extra_body):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    body.update(extra_body)
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    n_chunks = 0
    usage_prompt = None
    usage_completion = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage_prompt = obj["usage"].get("prompt_tokens", usage_prompt)
                usage_completion = obj["usage"].get("completion_tokens", usage_completion)
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if not piece:
                continue
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            n_chunks += 1
    if t_first is None:
        raise RuntimeError("nessun token generato (stream vuoto)")
    n_prompt_real = usage_prompt or n_prompt
    if usage_completion is None:
        raise RuntimeError("no usage.completion_tokens in the stream: the server ignores "
                           "stream_options.include_usage, decode tok/s cannot be measured honestly")
    prefill_s = t_first - t0
    decode_s = (t_last - t_first) if t_last and t_last > t_first else 0.0
    return {
        "prompt_tokens": n_prompt_real,
        "prompt_tokens_local": n_prompt,
        "gen_tokens": usage_completion,
        "gen_chunks": n_chunks,
        "prefill_s": round(prefill_s, 3),
        "prefill_tok_s": round(n_prompt_real / prefill_s, 1) if prefill_s > 0 else None,
        "decode_s": round(decode_s, 3),
        # tokens after the first chunk / time after the first chunk. The first chunk may carry
        # more than one token under speculation, so this is a slight UNDER-estimate, never an over.
        "decode_tok_s": round((usage_completion - 1) / decode_s, 2) if decode_s > 0 and usage_completion > 1 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30021")
    ap.add_argument("--model", default="deepseek-v4-flash-vision-exp")
    ap.add_argument("--tokenizer", required=True, help="dir con tokenizer.json del modello servito")
    ap.add_argument("--ctx", type=int, nargs="+", default=[100000, 250000])
    ap.add_argument("--gen", type=int, default=256, help="max token da generare per misurare il decode (il prompt chiede ~200 parole)")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None, help="file JSON dove appendere i risultati")
    ap.add_argument("--extra-body", default="{}", help='JSON extra per il body, es. \'{"chat_template_kwargs":{"thinking":false}}\'')
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.expanduser(args.tokenizer), trust_remote_code=True)
    extra = json.loads(args.extra_body)
    salt = f"{args.tag}-{int(time.time())}"

    results = []
    for target in args.ctx:
        prompt, n = build_prompt(tok, target, f"{salt}-{target}")
        print(f"[{args.tag}] ctx {target}: prompt costruito = {n} token, invio...", flush=True)
        try:
            r = run_one(args.url, args.model, prompt, n, args.gen, args.timeout, extra)
        except Exception as exc:  # noqa: BLE001
            print(f"[{args.tag}] ctx {target}: FALLITO — {exc}", file=sys.stderr, flush=True)
            results.append({"tag": args.tag, "ctx_target": target, "error": str(exc)})
            continue
        r.update({"tag": args.tag, "ctx_target": target, "url": args.url, "model": args.model})
        results.append(r)
        print(
            f"[{args.tag}] ctx {target}: prefill {r['prefill_tok_s']} tok/s "
            f"({r['prompt_tokens']} tok in {r['prefill_s']}s) · "
            f"decode {r['decode_tok_s']} tok/s ({r['gen_tokens']} tok in {r['decode_s']}s, {r['gen_chunks']} chunks)",
            flush=True,
        )

    if args.out:
        with open(os.path.expanduser(args.out), "a") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")
        print(f"[{args.tag}] scritto in {args.out}", flush=True)


if __name__ == "__main__":
    main()
