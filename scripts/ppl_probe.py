#!/usr/bin/env python3
"""Perplexity probe via vLLM /v1/completions prompt_logprobs. Same texts -> comparable NLL across engines."""
import json, sys, time, urllib.request, math, datetime

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30021"
model = sys.argv[2] if len(sys.argv) > 2 else None
out = sys.argv[3] if len(sys.argv) > 3 else None

TEXTS = {
 "it_prosa_1": "La mattina il paese si svegliava lentamente. Le serrande dei negozi si alzavano una dopo l'altra, e il profumo del pane appena sfornato attraversava la piazza fino alla fontana. Mia nonna diceva sempre che il silenzio delle sette era il momento migliore per pensare, prima che le voci dei bambini e il rumore dei motorini riempissero ogni angolo della giornata. Io non le credevo, allora; oggi mi accorgo che aveva ragione su quasi tutto.",
 "it_prosa_2": "Il problema principale della gestione dei rifiuti nelle piccole isole non è tecnico ma logistico: i costi di trasporto verso gli impianti della terraferma superano spesso il valore dei materiali recuperati, e nei mesi estivi la popolazione triplica senza che i servizi crescano di conseguenza. Le soluzioni proposte negli ultimi anni, dal compostaggio di comunità alla tariffazione puntuale, funzionano solo se accompagnate da controlli reali e da una comunicazione semplice, ripetuta e credibile.",
 "it_tecnico": "Per configurare il servizio è necessario modificare il file di configurazione principale, impostare la porta di ascolto e riavviare il demone. Se il riavvio fallisce, controllare il registro di sistema con il comando journalctl e verificare che nessun altro processo occupi la stessa porta. In ambienti con memoria unificata conviene inoltre svuotare la cache delle pagine prima di caricare modelli di grandi dimensioni.",
 "en_prose_1": "The harbor was quiet at that hour, the kind of quiet that comes after a storm has passed and before the fishermen return. She walked along the wet stones, counting the boats she recognized and noticing the ones she did not. Her father used to say that a town could be read from its harbor the way a face could be read from its eyes, and she had spent most of her life trying to prove him wrong.",
 "en_prose_2": "Most arguments about productivity confuse two different things: doing more work, and doing the right work. The first is a matter of energy and discipline; the second is a matter of judgment, which cannot be scheduled. People who are very good at the first often mistake it for the second, and organizations reward them for it, right up until the moment the environment changes and nobody notices.",
 "en_tech": "Speculative decoding runs a small draft model to propose several tokens, then verifies them in a single forward pass of the large target model. When the acceptance rate is high, the target model produces multiple tokens per step at roughly the cost of one. On memory-bandwidth-bound hardware this can double throughput, but the gain collapses when the draft disagrees with the target, which is typical for free-form prose in languages the draft saw rarely during training.",
 "code_py": "def merge_intervals(intervals):\n    intervals = sorted(intervals, key=lambda x: x[0])\n    merged = []\n    for start, end in intervals:\n        if merged and start <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n    return merged\n\n\ndef test_merge_intervals():\n    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]\n    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]\n",
 "code_bash": "#!/usr/bin/env bash\nset -euo pipefail\nfor f in \"$@\"; do\n  if [[ ! -f \"$f\" ]]; then\n    echo \"missing: $f\" >&2\n    continue\n  fi\n  sha256sum \"$f\" | awk '{print $1}' > \"$f.sha256\"\ndone\n",
}

def models():
    with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as r:
        return [m["id"] for m in json.load(r)["data"]]

def nll(text):
    body = json.dumps({"model": model, "prompt": text, "max_tokens": 1, "temperature": 0,
                       "prompt_logprobs": 0, "logprobs": 0}).encode()
    req = urllib.request.Request(f"{base}/v1/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        res = json.load(r)
    dt = time.time() - t0
    pl = res["choices"][0].get("prompt_logprobs")
    if pl is None:
        raise SystemExit("server returned no prompt_logprobs: " + json.dumps(res)[:500])
    lps = []
    for entry in pl[1:]:  # first token has no logprob
        if not entry:
            continue
        # entry: {token_id: {"logprob":..., "rank":..., "decoded_token":...}} -> the actual token has rank key; take max logprob among entries with rank (actual token is always included)
        vals = list(entry.values())
        # actual prompt token is the one included regardless of top-k; with prompt_logprobs=0 only it is present
        lps.append(vals[0]["logprob"] if len(vals) == 1 else max(v["logprob"] for v in vals))
    return -sum(lps) / len(lps), len(lps), dt

if model is None:
    model = models()[0]
results = {"base": base, "model": model, "when": datetime.datetime.now().isoformat(timespec="seconds"), "texts": {}}
tot_nll = 0.0; tot_n = 0
for k, t in TEXTS.items():
    m, n, dt = nll(t)
    results["texts"][k] = {"nll": round(m, 4), "ppl": round(math.exp(m), 3), "tokens": n, "sec": round(dt, 2)}
    tot_nll += m * n; tot_n += n
    print(f"{k:12s} nll={m:.4f} ppl={math.exp(m):8.3f} tok={n:4d} {dt:5.1f}s", flush=True)
results["mean_nll"] = round(tot_nll / tot_n, 4); results["mean_ppl"] = round(math.exp(tot_nll / tot_n), 3)
print(f"MEAN         nll={results['mean_nll']:.4f} ppl={results['mean_ppl']:.3f} model={model}")
if out:
    json.dump(results, open(out, "w"), indent=1)
