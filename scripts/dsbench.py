#!/usr/bin/env python3
"""Controlled A/B bench for the dsflash vLLM engine (direct :30021).
tok/s = verify_steps/s x tau; both are reported separately (Gaël's spec-measure.py idea) using vllm:spec_decode_* deltas.
Tasks: count (T=0, thinking off, fixed output), code (T=0, thinking off), prose (T=1.0, thinking on, effort max = live-like).
Usage: dsbench.py TAG [--tasks count,code,prose] [--runs 3] [--url http://127.0.0.1:30021]
Appends JSON lines to $DSBENCH_LOG (default ./dsbench.jsonl)
"""
import json, os, sys, time, argparse, urllib.request, statistics as st, datetime
ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--tasks", default="count,code,prose")
ap.add_argument("--runs", type=int, default=3); ap.add_argument("--url", default="http://127.0.0.1:30021")
ap.add_argument("--model", default="deepseek-v4-flash-0731"); ap.add_argument("--max-tokens", type=int, default=600)
ap.add_argument("--temp-prose", type=float, default=1.0); ap.add_argument("--effort", default="max")
a = ap.parse_args()
FILLER = ("The storage subsystem coordinator reconciles pending write intents against the durable manifest before "
          "acknowledging the commit, so a crash between the intent log and the manifest flush cannot surface a torn "
          "record to a reader that arrives afterwards. ")
TASKS = {
 "count": dict(prompt="Count from 1 to 220. One number per line. Output nothing else.", temp=0.0, thinking=False, filler=6),
 "code":  dict(prompt="Write a complete Python module implementing an LRU cache with per-entry TTL (class LRUCacheTTL with get/put/expire), plus pytest unit tests covering eviction order, TTL expiry and capacity 1. Output only code.", temp=0.0, thinking=False, filler=6),
 "prose": dict(prompt="Explain carefully why the daytime sky is blue and sunsets are red, covering Rayleigh scattering's wavelength dependence, the role of the eye's sensitivity, why the sky is not violet, and how aerosols and altitude change the effect. Then give two everyday experiments that demonstrate it.", temp=a.temp_prose, thinking=True, filler=6),
}
def metrics():
    t = urllib.request.urlopen(a.url + "/metrics", timeout=20).read().decode(); o = {}
    for ln in t.splitlines():
        if ln.startswith("vllm:spec_decode_num_") or ln.startswith("vllm:generation_tokens_total") or ln.startswith("vllm:prompt_tokens_total"):
            k, v = ln.split()[0], float(ln.split()[1]); o[k.split("{")[0]] = v
    return o
def one(task, i):
    T = TASKS[task]
    p = ("SALT-%s-%s-%d\n" % (a.tag, task, time.time_ns())) + FILLER * T["filler"] + "\n\n" + T["prompt"]
    body = {"model": a.model, "messages": [{"role": "user", "content": p}], "max_tokens": a.max_tokens,
            "temperature": T["temp"], "top_p": 1.0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": ({"thinking": True, "reasoning_effort": a.effort} if T["thinking"] else {"thinking": False})}
    m0 = metrics(); t0 = time.perf_counter(); tf = None; n = 0; usage = None; finish = None
    req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data: ") or ln == "data: [DONE]": continue
            d = json.loads(ln[6:])
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            if ch:
                dl = ch[0].get("delta") or {}
                if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning"):
                    if tf is None: tf = time.perf_counter()
                    n += 1
                if ch[0].get("finish_reason"): finish = ch[0]["finish_reason"]
    t1 = time.perf_counter(); m1 = metrics()
    g = lambda k: m1.get(k, 0) - m0.get(k, 0)
    dd, da, ds = g("vllm:spec_decode_num_draft_tokens_total"), g("vllm:spec_decode_num_accepted_tokens_total"), g("vllm:spec_decode_num_drafts_total")
    gen = (usage or {}).get("completion_tokens") or g("vllm:generation_tokens_total"); ptok = (usage or {}).get("prompt_tokens") or g("vllm:prompt_tokens_total")
    dt = (t1 - tf) if tf else 0.0
    return {"tag": a.tag, "task": task, "run": i, "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_tokens": ptok, "gen_tokens": gen, "ttft_s": round((tf - t0), 3) if tf else None,
            "prefill_tps": round(ptok / (tf - t0), 1) if tf and ptok else None,
            "dec_tps": round((gen - 1) / dt, 2) if dt > 0 and gen else None,
            "tau": round(1 + da / ds, 3) if ds else None, "alpha": round(da / dd, 3) if dd else None,
            "steps_s": round(ds / dt, 2) if dt > 0 and ds else None, "finish": finish}
rows = []
for task in a.tasks.split(","):
    for i in range(a.runs):
        r = one(task, i); rows.append(r)
        print("  %-5s run%d  gen=%-4s ttft=%5ss pre=%6s t/s  dec=%6s t/s  tau=%-6s steps/s=%-6s fin=%s" % (
            task, i, r["gen_tokens"], r["ttft_s"], r["prefill_tps"], r["dec_tps"], r["tau"], r["steps_s"], r["finish"]), flush=True)
        with open(os.path.expanduser(os.environ.get("DSBENCH_LOG", "dsbench.jsonl")), "a") as f: f.write(json.dumps(r) + "\n")
for task in a.tasks.split(","):
    rr = [x for x in rows if x["task"] == task and x["dec_tps"]]
    if rr:
        med = lambda k: st.median([x[k] for x in rr if x[k] is not None])
        print("%-10s %-5s dec med=%.2f (%.2f-%.2f)  tau med=%.3f  steps/s med=%.2f  prefill med=%.0f  n=%d" % (
            a.tag, task, med("dec_tps"), min(x["dec_tps"] for x in rr), max(x["dec_tps"] for x in rr), med("tau"), med("steps_s"), med("prefill_tps") or 0, len(rr)))
