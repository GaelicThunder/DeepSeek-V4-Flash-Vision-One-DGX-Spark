# TO_FIX — known issues and workarounds in use

| issue | status / workaround |
|---|---|
| **Kalibrated with 26 promoted layers does not boot at 245k context** (`3.91 GiB KV cache is needed ... 0.63 GiB available`) | 22 layers at `UTIL=0.925` (KV 4.29 GiB); never raise `UTIL` above 0.93 on a 128 GB Spark (hard-lock) |
| **`)Skip` (token 83480) appears at clause ends**, including inside tool-call arguments and reasoning | decode-path logits fault, not sampling: prefill scores it p 0.007 / rank 7, decode p 0.65 on the same prefix. Mask it — `"bad_words": [")Skip", ",Skip", ".Skip"]` per request, or `scripts/badwords_proxy.py` in front. Root cause still open: A/B the decode-only kernels (nvfp4 KV read-back, `B12X_MLA_SPARSE`, small-batch MoE). Evidence and detector in [docs/DECODE_PATH_TOKEN_LEAK.md](docs/DECODE_PATH_TOKEN_LEAK.md) |
| **Needle timings are prefix-cache contaminated** (filler not salted, 96 % hit rate) | correctness kept, timings ignored; prefill measured with `scripts/bench-ctx.py` instead |
| **One request at a time** (`Running: 1 reqs, Waiting: N`) | by design with the DSpark draft (`MAX_NUM_SEQS=1`); concurrent clients queue |
| **First image request after boot is slow** | one-time Triton/TileLang JIT for the encoder shapes; subsequent requests 3–4 s for a 400-token image |
| **`--mm-processor-cache-gb 0` and `--mm-encoder-attn-backend TORCH_SDPA` are mandatory** | defaults in `scripts/serve.sh`; the image has no FA2 build and the 4 GiB processor cache does not fit |
| **Boot fails silently after "loading" if the vision loader materializes the checkpoint** | fixed in `overlay/dsvision/vllm/models/deepseek_v4/nvidia/vl_model.py`; if you see `MemAvailable` fall to zero at the end of the load, an overlay file is not mounted |
| **`trellis_bits must be one of 3, 4, 5, 6; got 2`** | a guard file is not mounted — `DRY_RUN=1 scripts/serve.sh` and check all six `sparkinfer`/`exl3.py` mounts |
| **`experts do not match the plan used to size TP MoE scratch`** | mixed-K scratch fix in `exl3.py` not mounted |
| **HTTP 400 "Failed to apply partial"** on image requests | `mm_preprocess.py` / `context.py` overlay not mounted (the fork hides the real exception without `context.py`) |
| **`sparkinfer_mhc_pre is served only by the fused Gram kernel`** | `rms_norm_eps` must be `1e-6`; `tools/make_config.py` forces it — regenerate `config.json` if edited by hand |
| **Model loading dies with `NV_ERR_NO_MEMORY` / box hard-locks** | page cache of a previous model: `DROP_CACHES=1 scripts/serve.sh`; never raise `UTIL` above 0.92 on a Spark. `scripts/ramwatch.sh` (started by `start.sh`) stops the container under a `MemAvailable` floor before the lock |
| **`hf download` stalls** on large shards | make sure `hf_xet` is installed and not disabled |
| **`MODE=plain` is rejected** | valid values are `dspark`, `mtp0`, `mtp2`, `mtp3` (`mtp0` = no speculation) |
| **On the reference machine an extra `speculator.py` (DSpark hidden-state harvest) is mounted** | unrelated local feature, inert unless `DSPARK_HARVEST_DIR` is set; not part of this repo, measurements were taken with it inert |
