# TODO

- **23–24 promoted layers at a shorter served context.** Each layer costs 0.75 GiB of KV pool; at 245k the pack stops
  at 22. The layer *choice* is settled by measurement (`docs/KALIBRATED.md` §5: head / middle / tail swaps against the
  four spares; the ranking holds, nothing to gain from shuffling the tail).
- **Retrain the DSpark draft against Kalibrated.** τ is 3.66 on code / 1.88 on prose; the target changed, the draft did not. A deeper draft does not help (depth 8 measured worse than 5, `docs/KALIBRATED.md` §6b) and a rebuild from this pack is a no-op (`mtp.*` tensors unchanged): only training moves τ.

- **Close the `)Skip` root cause.** The mask in `scripts/badwords_proxy.py` hides it; it does not fix it. `MODE=mtp0`
  reproduces it (so the verify block is out); the indexer backend and 16-bit MoE activations leave it bit-identical;
  the KV-cache dtype cannot be switched on this stack (`fp8_ds_mla` fails on the page stride, §6b of
  `docs/KALIBRATED.md`). What is left: the `B12X_MLA_SPARSE` decode path itself and the small-batch MoE, one at a time,
  re-running `tools/token_leak_probe.py --prefix-text ... --verify-fix` until the decode probability matches the
  prefill one. Also worth a pass with `VLLM_COMPUTE_NANS_IN_LOGITS=1`.
- **Retrain the DSpark draft against this target.** τ is 3.2 on code vs 4.1 for the 3-bit recipe with a draft built
  the same way; the engine itself is faster (11.0 vs 10.3 verify steps/s). A draft that has seen the 256-expert 2-bit
  target's distribution should close most of the 35→46 tok/s gap.
- **Multi-image prompts and tool calls with images** — untested. The vision-prefill row-width issue vcruz305 patched in
  FlashInfer does not apply to this attention path in theory; check in practice with several images per prompt.
- **Salt the needle filler** so its timings become usable, then drop the separate long-context timing run.
- **Try `UTIL=0.90`.** 0.88 leaves ~9 GB of host headroom at 245k; the 3-bit recipe runs at 0.92–0.925 with less
  headroom and survives. More KV pool, or a longer `CTX`.
- **Upstream.** The overlay is a port onto a fork that will eventually rebase past PR #54566; at that point the vLLM
  half of the overlay should shrink to the unified-memory `load_weights` fix and the CUDA-graph-safe router, which
  are worth proposing upstream on their own.
- **Text-only view as a served option** (`tools/use_vision.sh off`) — it boots and was the debugging path, but it has
  not been benchmarked separately since the vision tensors were added.
- Nightly/weekly re-run of `scripts/bench/bench_all.py` when the image digest or the pack revision changes.
