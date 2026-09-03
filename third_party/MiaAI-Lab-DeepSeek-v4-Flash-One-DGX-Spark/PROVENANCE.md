# MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark — vendored files

Source: https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark
Commit: fdcd538fbf95fb15b2d6850db9613d22b2c889b8 ("Credit EXL3 to turboderp and BrandonMusicKy; limit 0xSero to the HF repo.")
License: MIT (LICENSE in this directory). Files are verbatim, unmodified.

| file | mounted at | why it is here |
|---|---|---|
| `image-patch/entrypoint-toolfix.sh` | `/patch-run-entrypoint.sh` | upgrades xgrammar at boot (tool-calling 500 in the 26.02 image) |
| `image-patch/entrypoint-256k.sh` | `/opt/recipe/scripts/entrypoint.sh` | 256k-context entrypoint: reads `KV_FP8_ROPE` / `VLLM_DSV4_PADDED_NVFP4` from env, builds the DSpark draft if missing |
| `image-patch/serve-ds4-flash.sh` | `/opt/vllm/serve-ds4-flash.sh` | the environment-driven vLLM launcher |
| `image-patch/coalesce_rank_sliced_exl3.py` | `/opt/recipe/scripts/coalesce_rank_sliced_exl3.py` | required by the entrypoint; not executed here (we ship tp1 directly) |
| `image-patch/build_dspark_draft.py` | — (used by `scripts/convert.sh`) | builds the K64 DSpark draft from the converted checkpoint |
| `image-patch/vllm/models/deepseek_v4/nvidia/dspark.py` | `/opt/vllm/vllm/models/deepseek_v4/nvidia/dspark.py` | DSpark draft model class |
| `image-patch/sparkinfer/moe/_shared/kernels/tiny_decode.py` | `/opt/sparkinfer/.../tiny_decode.py` | M<=4 decode kernel used by the draft |
| `image-patch/sparkinfer/attention/_shared/mla/prefill.py`, `prefill_mg.py` | `/opt/sparkinfer/.../mla/` | SM120 sparse-MLA prefill dispatcher + kernel |

Their recipe serves the REAP-pruned 3-bit text model (0xSero/deepseek-v4-flash-0731-spark). This
repository reuses their runtime layer unchanged and adds the vision model on top.
