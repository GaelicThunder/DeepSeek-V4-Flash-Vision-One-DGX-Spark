# TODO

- **Publish the MixedK+ pack.** The ready-to-serve directory (tp1/ + dspark-draft-k64/, ~100 GiB; or just the 22
  promoted 3-bit layer files, 53 GB) is only on the reference machine. Once it is on the Hub, `PLUS_REPO=<repo>
  ./start.sh` is the whole install; until then `scripts/aplus/` rebuilds it (2×H200, ~4 h; ~20 h on the Spark).
- **Choose the 22 layers by measurement, not by proxy.** The current set is the proxy-error ranking; swapping blocks
  of layers (25 min per boot + NLL run) can only improve it. Also test 23–24 layers at a shorter served context.
- **Retrain the DSpark draft against A+.** τ is 3.66 on code / 1.88 on prose; the target changed, the draft did not.

- **Close the `)Skip` root cause.** The mask in `scripts/badwords_proxy.py` hides it; it does not fix it. One boot
  with `MODE=mtp0` says whether the corruption needs the speculative verify block at all, then swap one decode-only
  kernel at a time (`kv_cache_dtype` off `nvfp4_ds_mla`, the `B12X_MLA_SPARSE` decode path, the small-batch MoE)
  and re-run `tools/token_leak_probe.py --prefix-text ... --verify-fix` until the decode probability matches the
  prefill one. Also worth a pass with `VLLM_COMPUTE_NANS_IN_LOGITS=1`.
- **Retrain the DSpark draft against this target.** τ is 3.2 on code vs 4.1 for the 3-bit recipe with a draft built
  the same way; the engine itself is faster (11.0 vs 10.3 verify steps/s). A draft that has seen the 256-expert 2-bit
  target's distribution should close most of the 35→46 tok/s gap.
- **Multi-image prompts and tool calls with images** — untested. The vision-prefill row-width issue vcruz305 patched in
  FlashInfer does not apply to this attention path in theory; check in practice with several images per prompt.
- **A RAM watchdog in the repo.** Serving on unified memory below ~1 GB `MemAvailable` can hard-lock the box; the one
  used during these measurements lives in the operator's tooling, not here. A 20-line `ramwatch.sh` that kills the
  container under a floor belongs in `scripts/`.
- **Salt the needle filler** so its timings become usable, then drop the separate long-context timing run.
- **Try `UTIL=0.90`.** 0.88 leaves ~9 GB of host headroom at 245k; the 3-bit recipe runs at 0.92–0.925 with less
  headroom and survives. More KV pool, or a longer `CTX`.
- **Upstream.** The overlay is a port onto a fork that will eventually rebase past PR #54566; at that point the vLLM
  half of the overlay should shrink to the unified-memory `load_weights` fix and the CUDA-graph-safe router, which
  are worth proposing upstream on their own.
- **Text-only view as a served option** (`tools/use_vision.sh off`) — it boots and was the debugging path, but it has
  not been benchmarked separately since the vision tensors were added.
- Nightly/weekly re-run of `scripts/bench/bench_all.py` when the image digest or the pack revision changes.
