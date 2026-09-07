# The `)Skip` artifact — one token that the decode path invents

**Short version.** Every so often a generation from this stack glues the token `)Skip` onto the end of a
clause: *"All 153 tests pass)Skip."*, *"…which is why it seems gone on your Linux box)Skip."*. It is one
vocabulary token (id **83480** in the DeepSeek-V4 tokenizer). It is not the sampler, not the temperature,
not the quantization, not the abliteration, and not the DSpark draft. **The model, asked to score that
exact context through the prefill path, gives that token a probability between 1e-6 and 1e-18.** The same
context regenerated one token at a time through the decode path puts it **first, at p 0.33–0.74** — the
same value on every repeat. So the logits row the sampler reads in the small-batch decode/verify pass is
wrong, and the sampler then emits it faithfully.

There is a one-line mitigation that needs no restart and costs nothing (`bad_words`), and an unfinished
root cause. Both are below. If you serve any DeepSeek-V4-Flash pack on this image and have never looked
for this, you probably have it — see [Detecting it](#detecting-it-on-your-own-box).

---

## What it looks like

The artifact is quiet. It does not crash anything, it does not produce garbage text, and it survives a
casual read because it looks like a typo. Counted over ~44 M characters of logged assistant output from
the reference machine:

| serving stack | assistant output | occurrences of `)Skip` | rate |
|---|---|---|---|
| llama.cpp, same model family | 2.6 M chars | **0** | — |
| this image + DSpark draft, sampler `temperature 1.0` / `top_p 1.0` | 43.4 M chars | 2,662 | ~1 per 4.3 k tokens |
| this image + DSpark draft, sampler `temperature 0.8` / `top_p 0.95` | 0.66 M chars | 8 | ~1 per 22 k tokens |

Two things make it worth fixing rather than ignoring:

- It lands **inside tool-call arguments**, not just prose. On the reference machine 51 tool calls carried
  it, which means it was written into real files: *"set the env to point at the host,Skip and run"*,
  *"…make the link point at something real or drop it.Skip"*.
- It lands **inside hidden reasoning**, where nobody proofreads it.

Two neighbours of the same token leak at about 4 % of the rate: `,Skip` (id 121099) and `.Skip`
(id 26104). They are the same fault, not legitimate text — *"so yes create works in this fastbootd,Skip
it needs the wipe first"*.

## What it is not

Each of these was excluded by measurement, not by argument.

- **Not the sampler.** It reproduces under pure greedy (`temperature 0`), where the standard rejection
  sampler can only emit the target's own argmax. Over 300 greedy tokens, zero emitted tokens failed to be
  rank-1 of their own reported top-5.
- **Not the speculative bookkeeping.** The `standard` rejection sampler in this image is textbook
  Leviathan and exact: accept iff `target_logprob > log(u) + draft_logprob`
  (`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:624`), recovered token by Gumbel-max over
  `log max(p−q,0)` (`:754-793`), bonus token from the target's own row K+1 (`:721-723`). The Gumbel noise
  is bounded to about **24 nats** (`vllm/v1/worker/gpu/sample/gumbel.py:14`), so no healthy logits row can
  emit a token ~39 nats below its maximum. The row itself has to be wrong.
- **Not the draft.** The DSpark draft directory has no `head.weight`, no `embed.weight` and no vocabulary
  map — it borrows the target's, so there is no id-remapping path to get wrong.
- **Not a constant/degenerate slot.** Probing the pathological prompts (empty, BOS only, BOS+BOS, EOS
  only, an empty user turn) never puts 83480 in the top-10.
- **Not the pack.** It appears on three different EXL3 packs from two different checkpoints — the 2-bit
  MixedK vision pack of this repo, a 3-bit REAP-pruned text pack, and a 3-bit REAP-pruned vision pack.

## The measurement that localizes it

Take a prompt, generate greedily until the artifact appears, then score **the identical prefix** two ways.

Prefix (80 tokens: a 30-token templated prompt plus 50 generated tokens ending in ` humility`):

| path | how it was obtained | p(`)Skip`) | top of the row |
|---|---|---|---|
| **prefill** | `/v1/completions` with the full 80 ids, `max_tokens 1` | **0.0069** (rank 7) | `.` 0.545, `.\n\n` 0.227 |
| **decode** | same, prefix cut by 1, `max_tokens 2` (so the slot is produced by the decode/verify pass) | **0.647** | `)Skip` 0.647, `.` 0.068 |
| **decode**, cut by 2 | `max_tokens 3` | **0.739** | `)Skip` 0.739, `.` 0.042 |

Properties of the shift:

- **Deterministic.** An exact repeat is bit-identical (`-0.4350` both times).
- **Order-preserving.** The rest of the top-5 keeps its order; only this one token moves, by +4.5 nats
  absolute and +6.6 to +7.4 nats relative to `.`.
- **Position-independent inside the verify block.** It leaks at block indices 1, 2 and 50 alike.
- **Prefix-sensitive.** Prepending a 6-token system prompt, or dropping the BOS, makes it vanish.

## Why this token

`head.weight` row 83480 is an outlier. Measured on both packs with a chunked memmap over the BF16 tensor:

| pack | row norm 83480 | median | p99 | max | rank |
|---|---|---|---|---|---|
| 2-bit MixedK vision (this repo) | **25.45** | 11.31 | 14.93 | 30.36 | **#28** / 129,280 |
| 3-bit REAP text | **25.19** | 11.25 | 14.82 | 29.86 | **#26** / 129,280 |

The two heads are not byte-identical (127,310 rows differ by more than 1e-3 in norm), so this is a
property of the **released DeepSeek-V4 head**, not of anyone's quantization. Its immediate neighbours in
the vocabulary are ordinary (11.4, 11.1, 11.9, 11.6) — it is an isolated spike. The head rows most
cosine-similar to it are exactly the family that leaks: `,Skip` 0.579, `).#` 0.392, `)import` 0.385,
`.Skip` 0.380.

So the mechanism is: a small, deterministic, prefix-dependent perturbation of the final hidden state in
the decode path projects onto a direction where this row is unusually long, and a few nats of logit
appear on this one token. At a clause end — where the true top mass is spread over `.` (0.55) and `.\n\n`
(0.23) — a few nats is enough to take the argmax. Mid-word, where the top token has p ≈ 1, it is not.
That is why the artifact only ever appears at clause boundaries.

**This part is not fully closed**, and the honest gaps are:

1. The measured shift (+4.5 to +7.4 nats) explains the case dissected above, but the extreme observations
   (prefill p = 1e-18) would need +40 nats, which would not preserve the top-5 order. Either the
   magnitude varies with the prefix, or those positions are a grosser corruption of the same row.
2. Twenty-seven head rows are *longer* than 83480 (the longest is a rare multilingual fragment) and never
   leak, so the perturbation has a specific direction that has not been identified.
3. It has not been tested with speculation off. That is one boot away and would say whether the fault
   lives in the K+1 verify rows or in any single-token decode step.

The remaining suspects are the decode-only kernels this image uses and llama.cpp does not: the
`nvfp4_ds_mla` KV read-back, the `B12X_MLA_SPARSE` decode path with chunk merge, and the small-batch MoE
path. The A/B is cheap — swap one at a time and re-run the contrast above until the decode number
returns to the prefill number.

## The fix that works today

`bad_words` is applied **in place on the very tensor the rejection kernels read**, for all K+1 verify rows,
before acceptance, recovery and the bonus token:

```
vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:131  processed_logits = self.sampler.apply_sampling_params(logits, ...)
vllm/v1/worker/gpu/sample/sampler.py:175                 self.bad_words_state.apply_bad_words(...)
vllm/v1/worker/gpu/sample/bad_words.py:164               tl.store(logits_ptr + ... + last_token, -float("inf"))
vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:139  rejection_sample(processed_logits, draft_logits, ...)
```

So masking the token makes the corrupted row fall back to its next-best entry, which is the `.` or `,`
that belongs there. Add it to any request:

```json
{ "model": "...", "messages": [...], "bad_words": [")Skip", ",Skip", ".Skip"] }
```

Verified on the reference machine on the repro prefix: `' humility)Skip'` → `' humility.'`.

Notes that save time:

- **`top_p` and `top_k` do not help.** In the corrupted row the token is already rank 1, so no nucleus
  excludes it.
- **`min_p` and `logit_bias` are rejected outright** under speculative decoding on this image
  (`HTTP 400: The min_p and logit_bias sampling parameters are not yet supported with speculative
  decoding`). `bad_words`, `top_k`, `top_p`, `temperature` and `repetition_penalty` are accepted.
- **It cannot be set server-side.** `generation_config.json` only carries the sampler whitelist
  (temperature, top_p, top_k, min_p, repetition_penalty, max_new_tokens); `bad_words` is not in it.
- **Banning the token does not ban the text.** `.Skip` is legitimate in C# (`.Skip(10)`); with the token
  masked the model writes `.` followed by `Skip`, which detokenizes to the same string.
- **A fine-tuned draft cannot fix it.** The draft only changes what is proposed; a correct verifier
  rejects the token whatever the draft says. This is a target-side fault.

If your client cannot set `bad_words`, [`scripts/badwords_proxy.py`](../scripts/badwords_proxy.py) is a
dependency-free reverse proxy that injects it into every request, streaming included.

## Detecting it on your own box

[`tools/token_leak_probe.py`](../tools/token_leak_probe.py) needs nothing but a served endpoint. It
generates greedily, then re-scores its own output through the prefill path and reports every emitted
token that prefill considers improbable — which finds this class of fault without knowing the token id in
advance:

```console
$ python3 tools/token_leak_probe.py --base http://127.0.0.1:30021 --model <served-name> \
    --prefix-text "$(cat prefix.txt)" -n 3 --verify-fix
generating 3 greedy tokens after the seed prefix ...
re-scoring the generation through the prefill path ...

  pos  emitted           decode p   prefill p   rank    ratio
   50  ')Skip'              0.647    6.86e-03      7      94x   <== SUSPECT
     context: ... quiet obsession. It is also a lesson in humility
     prefill wanted: '.' 0.545, '.\n\n' 0.227, ' and' 0.045

1 suspect token in 52 (1.92 %). Mitigate with: "bad_words": [")Skip"]

replaying each slot through the DECODE path (prefix cut by one, two tokens):
  pos 50   stock=' humility)Skip'   masked=' humility.'   REPRODUCED, and the mask removes it
```

Without `--prefix-text` it generates 400 greedy tokens of its own and scores those, which is the
way to find out whether you have the problem at all. A flagged token must satisfy three
conditions — the model was confident, prefill disagrees, and the gap is at least 20× — because
ordinary prefill/decode numerical drift moves a flat distribution by about a factor of two and
would otherwise drown the signal. `--verify-fix` then replays each flagged slot through the
decode path and says plainly whether it reproduced.

`--head <path-to-tp1>` additionally reports the outlier rows of `head.weight`, which is where to look if
your leaked token is a different one.

## Status 2026-09-05

- Reproduces with speculation off (`MODE=mtp0`): p 0.27 at the seeded slot. The verify block and the sampler are out.
- `INDEXER_BACKEND=native` and `BACKEND=b12x-a16` leave the seeded-slot probability bit-identical (0.6473): the fault
  is in the part of the decode path those knobs do not touch (candidates: the small-batch MoE decode kernel, the
  decode-side head GEMV path). A free 400-token run is not a test — it reports "clean" whenever the trajectory never
  wanders into the trap; always force the slot.
- Mitigation unchanged: `bad_words` (`scripts/badwords_proxy.py`); the reference deployment applies it at its gateway.
