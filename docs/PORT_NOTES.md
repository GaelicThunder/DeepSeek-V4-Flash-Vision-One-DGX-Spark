# Port notes — vLLM PR #54566 onto the sparkinfer image

The image `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer` (digest pinned in `scripts/serve.sh`) ships
the NVIDIA 26.02 vLLM fork (`vllm-dev-e0593cdf`, torch 2.12+cu130, Python 3.12) plus NVIDIA's `sparkinfer`
kernels under `/opt/sparkinfer`. Upstream vLLM merged DeepSeek-V4-Flash-Vision-Exp support on 2026-09-02
(PR #54566, merge commit `1356635d837c4ef002ec98c1a0296e7ff60be3c1`, +2918/−152, new architecture class
`DeepseekV4ForConditionalGeneration`). The fork in the image predates it by about a month and has diverged in
the multimodal processor API and a few signatures. This document lists what had to change to make the PR run there,
in the order the blockers appeared. The upstream hunks as extracted are in `reference/pr54566/hunks.txt`.

The overlay is mounted read-only over the image's own files (`scripts/serve.sh`), so each file below replaces
its namesake in `/opt/vllm/vllm/…` or `/opt/sparkinfer/sparkinfer/…`.

## Layer 0 — make 2-bit legal (sparkinfer + exl3.py)

The pack is 2-bit trellis with six 3-bit layers (`bitrates.json`); the image guards `K ∈ {3,4,5,6}` in six places.
`tools/apply_k2_patch.py` performs five of them as exact-match edits and is how the overlay files were produced:

| file | edit |
|---|---|
| `sparkinfer/_lib/intrinsics.py` | two guards `(3,4,5,6)` → `(2,3,4,5,6)` |
| `sparkinfer/moe/fused_moe/_impl.py` | `trellis_bits` guard |
| `sparkinfer/moe/_shared/kernels/w4a16/kernel.py` | `_TRELLIS256_BITS` |
| `sparkinfer/moe/_shared/kernels/w4a16/prepare.py` | two guards |
| `vllm/model_executor/layers/quantization/exl3.py` | tier check + rank-sliced bitrate check |

The sixth was found at boot: `sparkinfer/moe/_shared/execution.py:303`, a dataclass `__post_init__`
(`trellis_bits must be one of 3, 4, 5, 6; got 2`). Widened the same way. The decode PTX is generic in the bit
width; only the guards changed.

**Mixed K needs per-layer scratch.** `exl3.py` kept one runtime cache entry — one MoE scratch arena — for all
layers. With K3 and K2 layers in the same model the second kind to run hit
`experts do not match the plan used to size TP MoE scratch`. The cache key now includes the layer's weight plan:

```python
plan_token = getattr(layer, "_exl3_plan_key", None)
if plan_token is None:
    _plan = layer.exl3_trellis_weights.plan
    try:
        hash(_plan); plan_token = _plan
    except TypeError:
        plan_token = repr(_plan)
    layer._exl3_plan_key = plan_token
key = (plan_token, ...)
```

**`rms_norm_eps`.** The fused mHC Gram kernel (`norm/mhc/_impl.py`) accepts only `1e-6`; the pack's config says
`1e-20`. `tools/make_config.py` forces `1e-6` (numerically irrelevant at BF16).

## Layer 1 — the vision port (vllm/)

New files dropped in from the PR: `models/deepseek_v4/common/vision.py` (ViT), `common/mm_preprocess.py`
(processor), `nvidia/vl_model.py` (the `ForConditionalGeneration` wrapper), `vl_stub.py`,
`model_executor/layers/fused_moe/router/dsv4_topk.py`. Patched to the PR's delta: `nvidia/model.py`
(`gate.bias_vl` parameter, `image_sentinel_lo`, gate bias on the three hash layers, both router call sites,
forward guard), `attention.py`, `fused_moe/layer.py`, `router/fused_topk_bias_router.py`, `router/router_factory.py`,
`model_executor/models/{registry,config}.py`, `config/model.py`, `transformers_utils/configs/deepseek_v4.py`,
`transformers_utils/model_arch_config_convertor.py`, `tokenizers/deepseek_v4_encoding.py` (`<｜deepseek_image｜>`
placeholder, `flatten_content_blocks`), `v1/worker/gpu_model_runner.py`, `multimodal/processing/context.py`,
`models/deepseek_v4/__init__.py`. Then, in order:

### 1. `ModelArchConfigConvertorBase.__init__() takes 3 positional arguments but 4 were given`
The PR's `DeepseekV4ModelArchConfigConvertor` passes `revision`; this base class has no such parameter.
The overlay's convertor takes `*args, **kwargs`. It rewrites `architectures` to the vision class when
`vision_n_layers > 0` (and not `_dsv4_vl_inner`), sets `is_mm_prefix_lm`, and registers under `"deepseek_v4"`.

### 2. `is_vit_use_data_parallel() takes 0 positional arguments but 1 was given`
Three call sites in `common/vision.py` go through a wrapper that tries the new signature and falls back:

```python
def _vit_data_parallel(num_heads: int) -> bool:
    try:
        return is_vit_use_data_parallel(num_heads)
    except TypeError:
        return is_vit_use_data_parallel()
```

### 3. `create_fused_moe_router() got an unexpected keyword argument 'bias_vl'`
`router_factory.py` gains the `bias_vl` / `image_sentinel_lo` passthrough, and the bias-router branch is selected
when `bias_vl` is present, not only when `e_score_correction_bias` is.

### 4. The fatal one: `vl_model.load_weights`
Upstream:

```python
loaded = loader.load_weights(sorted(self.hf_to_vllm_mapper.apply(weights), key=...))
```

`sorted()` consumes the generator. With the image's InstantTensor loader in BUFFERED mode every yielded tensor is a
real buffer, so this asks for roughly twice the checkpoint — for an 88 GiB model on a 122 GiB box that is
`MemAvailable 118 → 0.5 GiB` and a silent death right after the last tensor, no traceback, eleven boots in a row.
The overlay streams the language model and defers only the vision side (~0.9 GiB):

```python
deferred: list[tuple[str, torch.Tensor]] = []

def _language_first():
    for name, w in self.hf_to_vllm_mapper.apply(weights):
        if name.startswith("language_model."):
            yield name, w
        else:
            deferred.append((name, w))

loader = AutoWeightsLoader(self)
loaded_params = loader.load_weights(_language_first())
if deferred:
    loaded_params |= loader.load_weights(iter(sorted(deferred, key=lambda x: x[0])))
```

Result: `Model loading took 83.58 GiB`, host memory flat, minimum `MemAvailable` during load 24 GB.
vcruz305 hit the same bug on PyPI vLLM the same day and fixed it with a shard-at-a-time generator.

### 5. `CUDA error: operation not permitted when stream is capturing`
Image tokens (five consecutive sentinel ids from `IMAGE_SENTINEL_BASE_ID`) must pick experts with `gate.bias_vl`
instead of the text bias / hash table. The straightforward implementation checks `mask.any()` and indexes with
`.nonzero()`; both are host syncs and abort CUDA-graph capture. The overlay version in `fused_topk_bias_router.py`
computes the alternative routing for every row and selects with `torch.where`, static shapes only:

```python
IMAGE_SENTINEL_SPAN = 5
def _reroute_image_tokens_(topk_weights, topk_ids, gating_output, input_tokens,
                           bias_vl, image_sentinel_lo, renormalize, routed_scaling_factor):
    ids = input_tokens.reshape(-1).long()
    mask = ((ids >= image_sentinel_lo) & (ids < image_sentinel_lo + IMAGE_SENTINEL_SPAN)).unsqueeze(1)
    scores = torch.sqrt(F.softplus(gating_output.float()))
    choice = scores + bias_vl.reshape(1, -1).float()
    _, sel = torch.topk(choice, k=topk_weights.shape[-1], dim=-1)
    w = scores.gather(1, sel)
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    w = (w * routed_scaling_factor).to(topk_weights.dtype)
    topk_weights.copy_(torch.where(mask, w, topk_weights))
    topk_ids.copy_(torch.where(mask, sel.to(topk_ids.dtype), topk_ids))
```

This is why the recipe keeps CUDA graphs (PIECEWISE + FULL captured at boot) where the PyPI route runs eager.

### 6. HTTP 400 `Failed to apply partial …` on any image request
The fork's `multimodal/processing/context.py` swallowed the inner exception. The overlay surfaces it:
`AttributeError: 'DeepseekV4VLProcessor' object has no attribute '_merge_kwargs'`, raised from
`call_hf_processor_mm_only`, a code path that does not exist upstream. Declaring the base class's documented
override routes around it:

```python
def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs) -> BatchFeature:
    out = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
    if "input_ids" not in out:
        tokenizer = self.info.get_tokenizer()
        out["input_ids"] = [list(tokenizer.encode(prompt))]
    return out
```

### 7. HTTP 500 `'input_ids'`
This fork pops `input_ids` from the processor's `BatchFeature`; the PR's processor never sets it. Hence the
second half of the override above.

### 8. `ImportError: cannot import name '_plan_prompt_updates'`
The PR's processor uses upstream's prompt-update planner, which the fork lacks. `common/dsv4_mm_compat.py` is an
AST-extracted backport of `_MatchedUpdate`, `_UpdateQueue`, `_QueueMatch`, `_target_key`,
`_compile_prompt_update_queues`, `_IterMatches`, `_find_queue_match`, `_next_priority`,
`_plan_prompt_updates_with`, `_plan_prompt_updates` (213 lines). `mm_preprocess.py` imports upstream's and falls
back to it.

### 9. `EngineDeadError` → `CUDA FlashAttention is unavailable: FA2: cannot import name '_vllm_fa2_C'`
The image was never built with FlashAttention-2. The vision encoder's attention backend is forced to
`TORCH_SDPA` (`--mm-encoder-attn-backend TORCH_SDPA`), and the multimodal processor cache is disabled
(`--mm-processor-cache-gb 0`; its 4 GiB default does not fit next to the weights). Both are defaults in
`scripts/serve.sh`.

## What is not in the overlay

- No change to any CUDA/PTX kernel. Trellis decode, MLA, DSA indexer and the mHC norms are the image's own.
- No change to the DSpark draft path (`nvidia/dspark.py` is MiaAI-Lab's, unmodified). The draft ignores
  `gate.bias_vl`, as upstream's does.
- vcruz305's SM120 wide-row prefill patch for FlashInfer is not needed on this path (encoder on SDPA, language model
  on sparkinfer MLA) and is not included.
