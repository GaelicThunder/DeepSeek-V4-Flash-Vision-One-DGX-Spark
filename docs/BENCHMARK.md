# Benchmark — 2-bit MixedK (256 experts) vs 3-bit REAP (216 experts)

All numbers from 2026-09-03 on the same machine (ASUS GX10 / DGX Spark GB10, 128 GB), one engine at a time,
same items, same grading code. Raw outputs in `receipts/`.

## Engines

| tag | what | serving |
|---|---|---|
| `dsvision` | this recipe: vcruz305 MixedK, 256 experts, K2 + six K3 layers, vision, abliteration baked into 26 `wo_b` tensors | sparkinfer image + overlay, `MODE=dspark`, util 0.88 (the first run of the day was at 0.86; quality is identical, deterministic) |
| `dsflash` | MiaAI-Lab recipe: 0xSero REAP-K216, uniform 3-bit, text | same image, `MODE=dspark`, util 0.925, runtime refusal-ablation hook on (`DSFLASH_ABLATE=1`, λ 3.5, layers 10–42) — its daily-driver default |
| `dsflash-noabl` | same as `dsflash` with the runtime ablation hook off | util 0.92 |

Both run at `max_model_len 245760`. `dsvision` was run twice (`bench-dsvision-20260903.json`,
`bench-dsvision-back-20260903.json`) on two separate boots, which gives the harness's own run-to-run noise.

## Items (frozen once, `scripts/bench/prep_data.py`)

- **MMLU-Pro** (`TIGER-Lab/MMLU-Pro`, test split): 18 questions from each of the 14 categories, seed `20260903`,
  251 items with ≥4 options (10 in nearly all). Scored on the **logprob of the answer letter** after
  `…\nAnswer:` — one forward pass, no generation, so no item can be lost to a token budget. Each item is asked
  twice: options in the original order and rotated by 3, which moves the correct letter. Reported: accuracy per
  order, mean, items correct under both orders, agreement between orders, mean logprob margin over the runner-up.
- **MATH-500** (`HuggingFaceH4/MATH-500`) level 5, first 60 by `unique_id`: free generation, thinking off,
  `max_tokens 2400`, temperature 0; the last `\boxed{}` is compared after LaTeX normalization
  (`\dfrac→\frac`, `\left/\right`, `\text{}`, spaces, degree signs, trailing `.0`).
- **Needle**: a code planted at depth 0.1 / 0.5 / 0.9 of a filler document sized to ~4k, ~32k and ~131k tokens
  (3,685 / 29,165 / 118,437 prompt tokens as tokenized), `max_tokens 24`, exact substring match.
  The filler is repetitive and not salted, so the server's prefix cache hit 96 %: **the pass/fail is valid, the
  timings are not** — prefill speed is measured separately with `scripts/bench-ctx.py`.
- **Perplexity** (`scripts/ppl_probe.py`): mean NLL over 8 fixed passages (Italian and English prose, technical
  prose, Python, bash) via `prompt_logprobs`.
- **Speed** (`scripts/dsbench.py`, medians of 3): three tasks — count 1..220 at T=0, a 600-token coding task at
  T=0, a physics explanation at T=1 with thinking on — reading `vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total`
  deltas so that τ (mean accepted length per verify step), α (per-draft-token acceptance) and verify steps/s are
  separated from tok/s.
- **Long context** (`scripts/bench-ctx.py`): prompts built with the model's tokenizer to 16,384 / 65,536 / 131,072
  tokens, each with a unique salt (no prefix-cache reuse), 128 generated tokens.

## Quality

| | dsvision | dsvision (2nd boot) | dsflash | dsflash-noabl |
|---|---|---|---|---|
| MMLU-Pro, original order | 161/251 · 64.14 % | 161/251 | 156/251 · 62.15 % | 154/251 · 61.35 % |
| MMLU-Pro, rotated by 3 | 162/251 · 64.54 % | 162/251 | 149/251 · 59.36 % | 148/251 · 58.96 % |
| **mean** | **64.34 %** | 64.34 % | **60.75 %** | **60.16 %** |
| correct under both orders | 134 | 133 | 123 | 119 |
| agreement between orders | 78.1 % | 77.3 % | 76.5 % | 74.5 % |
| mean margin over runner-up (nat) | 2.69 / 2.43 | 2.69 / 2.43 | 2.47 / 2.30 | 2.42 / 2.15 |
| MATH-500 L5 | 50/60 (3 truncated, 792 tok mean) | 49/60 (5 truncated, 863 tok) | 50/60 (4 truncated, 757 tok) | 48/60 (2 truncated, 715 tok) |
| needle 9 positions | 9/9 | 9/9 | 9/9 | not run |
| perplexity NLL / PPL | 1.5140 / 4.545 | 1.5135 / 4.543 | 1.6814 / 5.373 | 1.6927 / 5.434 |

Paired analysis, dsvision (first boot) vs dsflash, 502 MMLU-Pro decisions:

| | both right | only dsvision | only dsflash | both wrong |
|---|---|---|---|---|
| original order | 135 | 26 | 21 | 69 |
| rotated | 132 | 30 | 17 | 72 |
| pooled | 267 | 56 | 38 | 141 |

Exact two-sided McNemar on the 94 discordant decisions: **p = 0.079** (0.56 on the original order alone, 0.079 on
the rotated order alone). MATH: 49 solved by both, `test_geometry_229` only by dsvision, `test_geometry_561`
only by dsflash, 9 by neither.

Per category (sum of both orders, out of 36 — 34 for computer science):

| category | dsvision | dsflash |   | category | dsvision | dsflash |
|---|---|---|---|---|---|---|
| biology | 32 | 28 |   | history | 18 | 16 |
| business | 22 | 21 |   | law | 16 | 18 |
| chemistry | 19 | 21 |   | math | 13 | 16 |
| computer science | 23 | 22 |   | other | 23 | 19 |
| economics | 26 | 26 |   | philosophy | 28 | 26 |
| engineering | 22 | 19 |   | physics | 22 | 17 |
| health | 28 | 25 |   | psychology | 31 | 31 |

With 36 decisions per cell these are noise-level differences; the total is what carries information.

### Reading it

1. The 2-bit MixedK pack with all 256 experts is **not less capable** than the 3-bit pack with 40 experts pruned:
   +3.6 points on MMLU-Pro, equal on MATH and needle, 15 % lower perplexity. The lead is at the edge of
   significance; "not worse" is the safe statement.
2. The runtime ablation hook on the 3-bit engine is not the explanation: turning it off moves 0.6 points, inside
   the noise, and slightly *raises* perplexity.
3. Harness noise, from repeating dsvision on a second boot: ±1 item on MATH, ±1 on robust-MMLU, 0 on the mean.
4. Perplexity reproduces to the third decimal across boots (1.5140 / 1.5135; 1.6814 on both days for dsflash), so
   the 4.54-vs-5.37 gap is a property of the packs, not of loading.

What this does **not** measure: a 256-expert 3-bit model (does not fit), long free-form generation quality,
tool calling, or anything about images beyond the probe.

## Speed

`scripts/dsbench.py`, medians of 3, single stream, `MODE=dspark` (K5 draft):

| task | dsvision dec tok/s | τ | steps/s | dsflash dec tok/s | τ | steps/s |
|---|---|---|---|---|---|---|
| count (T=0, thinking off) | 36.03 (34.96–37.35) | 3.259 | 11.07 | 46.12 (45.77–46.41) | 4.444 | 10.36 |
| code (T=0, thinking off) | 35.26 (34.85–37.67) | 3.226 | 10.94 | 41.79 (40.20–45.23) | 4.103 | 10.19 |
| prose (T=1, thinking on) | 23.25 (21.51–23.99) | 2.109 | 11.02 | 21.10 (21.04–21.64) | 2.048 | 10.32 |

Prefill in these runs (short prompts, 500–600 tok/s) is dominated by launch overhead; see the long-context table.

`scripts/bench-ctx.py`, salted prompts, 128 generated tokens after the prompt:

| prompt tokens | dsvision prefill | dsvision decode | dsflash prefill | dsflash decode |
|---|---|---|---|---|
| 16,375 | 1,239.2 tok/s (13.2 s) | 10.68 | 1,178.8 (13.9 s) | 10.20 |
| 65,515 | 1,197.5 (54.7 s) | 11.03 | 1,201.5 (54.5 s) | 10.71 |
| 130,983 | 1,130.4 (115.9 s) | 11.40 | 1,131.5 (115.8 s) | 11.33 |

Decode here is after a long, unpredictable prompt (the draft accepts little), which is why it is far below the
`dsbench` numbers for both engines; compare within the table, not across.

Older single-stream probe (`scripts/speed_probe.py`, 300 tokens including prefill): dsvision 33.4 / 32.4 tok/s
code, 17.0 / 16.9 prose; dsflash 47.5 / 47.4 code, 15.4 / 14.7 prose — consistent with `dsbench`.

## Reproducing

```bash
pip install datasets                                    # once
python3 scripts/bench/prep_data.py                      # regenerates items_mc.json / items_math.json (same seed)
python3 scripts/bench/bench_all.py out.json --base http://127.0.0.1:30021 --only mc,math,needle --conc 6
python3 scripts/ppl_probe.py   http://127.0.0.1:30021 deepseek-v4-flash-vision-exp ppl.json
python3 scripts/dsbench.py TAG --url http://127.0.0.1:30021 --model deepseek-v4-flash-vision-exp --runs 3
python3 scripts/bench-ctx.py --url http://127.0.0.1:30021 --model deepseek-v4-flash-vision-exp \
        --tokenizer ~/models/deepseek-v4-flash-vision-spark/tp1 --ctx 16384 65536 131072 --gen 128 \
        --extra-body '{"chat_template_kwargs":{"thinking":false}}'
```

`--conc 6` only parallelizes the request submission; with the draft on, vLLM serves one sequence at a time.
The MC part takes ~2 min per order, MATH ~20–25 min, needle ~6 min. `dsbench.py` appends to
`./dsbench.jsonl` (override with `DSBENCH_LOG`).
