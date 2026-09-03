# Conversion — MixedK pack → rank-sliced tp1

`scripts/convert.sh` runs the whole thing; this is what it does and why.

## The two layouts

**Source** — `vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK`: 48 HF shards (`model-000NN-of-00048.safetensors`),
~95 GB. Routed experts are EXL3 trellis tensors (`layers.L.ffn.experts.E.wP.{trellis,suh,svh,mcg}`; MCG codebook)
at K2, with layers 3, 13, 21, 22, 28, 41 at K3. Everything non-routed (attention projections, dense early layers,
shared experts, DSA indexer, norms, embeddings, `lm_head`, the vision tower and aligner, the three DSpark/MTP stages)
is BF16 as in the original checkpoint, except the MTP routed experts which stay in their source MXFP4 format.
The abliteration is 26 swapped `layers.10-35.attn.wo_b.weight` tensors (drowzeys, λ 3.5); every expert is
byte-identical to the non-abliterated MixedK.

**Target** — the layout `0xSero/deepseek-v4-flash-0731-spark` uses and the sparkinfer image loads:

| file | contents |
|---|---|
| `exl3-layer-LLL-tp1-rank0.safetensors` (43) | routed experts of layer L, names `layers.L.ffn.experts.E.wP.rank0.{trellis,suh,svh,mcg}`, `mcg` as a 0-d tensor |
| `carried-00N.safetensors` | all non-expert tensors; BF16 projections stored as FP8 e4m3 with `ue8m0` 128×128 block scales (`.scale`, `F8_E8M0`) wherever the reference has them |
| `carried-vis-00N.safetensors` | added by `add_vision_tensors.py`: `vision.*`, `aligner.*`, `image_*`, 46 × `layers.N.ffn.gate.bias_vl` (F32[256]), the hash-layer `layers.{0,1,2}.ffn.gate.bias` |
| `model.safetensors.index.json` | weight map (138,681 tensors in the vision view, 138,365 text-only) |
| `rank-sliced-tp1-manifest.json` | file list with sha256 and sizes; the entrypoint verifies it (`VERIFY_MODEL_CHECKSUMS`) |
| `bitrates.json` | per layer: `routed` K for each of the 256 experts |
| `config.json`, `quantization_config.json` | `hybrid_tr3_tail.bits = "mixed"`, `k_values [2,3]`, `bits_per_expert = "bitrates.json:routed"`, `quant_method exl3`, `layer_bits` |

## Steps

1. **`tools/convert_vision_pack.py --src PACK --ref-spec reference/ref_spec.json --out tp1`**
   Reads shard headers, groups tensors per layer, copies each expert's trellis payload unchanged (never decoded),
   renames to the tp1 scheme, and requantizes BF16 projections to FP8 + ue8m0 exactly where `ref_spec.json` says
   the reference has a `.scale` tensor. `ref_spec.json` is the dtype/shape/scale map of the 0xSero tp1 checkpoint
   (`tools/ref_spec.py` produced it from that checkpoint's headers) and is vendored so nobody needs the 99 GB
   reference. Requantization is exact for values that were FP8 in the original checkpoint; the 26 abliterated
   `wo_b` tensors are genuinely BF16 and show ~3 % relative L2 error after FP8 rounding (`verify_tp1.py` reports it).
   MTP experts 216–255 have no counterpart in the REAP-K216 reference; the converter uses expert 0 of the same
   MTP stage as the template (`refspec()` fallback). Reads are sequential in shard offset order — fine from a USB disk,
   ~25 min. Peak RAM ~2.3 GiB (one layer's expert span).
2. **`tools/make_config.py PACK/config.json reference/0xsero-tp1-config.json tp1 tp1/bitrates.json`**
   The pack's config is the architecture truth for Vision-Exp; the reference config supplies the rank-sliced metadata
   blocks the image expects. Writes the text-only view (vision keys dropped) and forces `rms_norm_eps 1e-6`.
3. **DSpark K64 draft** — `third_party/…/build_dspark_draft.py --plan reference/draft_plan.json`. The image's
   builder expects a REAP plan; `tools/make_draft_plan.py` synthesized one for the unpruned checkpoint whose ranking is
   the 64 original expert ids the 0731 draft selected, so the draft keeps the same experts as the text recipe.
   Built from the text view, before the vision tensors are appended. ~3 GB.
4. **Vision view** — copies of `config.json`, the index and the manifest are kept as `*.text.json`;
   `tools/add_vision_tensors.py` appends the 316 vision-side tensors verbatim into `carried-vis-*.safetensors`
   and updates the vision index + manifest; `tools/make_vision_config.py` writes `config-vision.json`
   (`DeepseekV4ForConditionalGeneration` + the eleven `vision_*` hyper-parameters from the pack).
   `tools/use_vision.sh on` makes the vision view current. `off` returns to text-only, which loads the same
   language model without the tower (the text view was how the K2/mixed-K work was debugged).
5. **`tools/verify_tp1.py PACK tp1 300`** — every tensor's name/dtype/shape against the source, byte equality on
   all small tensors plus 300 sampled experts, FP8 dequant error on the requantized ones.

On disk afterwards: `tp1/` 89 GB, `dspark-draft-k64/` 3.0 GB. The source pack can be deleted.

## Numbers that come out of the conversion

- index `total_size` 87.75 GiB before the vision tensors, +0.9 GiB after;
- `Model loading took 83.58 GiB` at boot;
- `bitrates.json`: 43 layers × 256 experts, six layers at K3 (`3, 13, 21, 22, 28, 41`);
- `verify_tp1.py`: names/dtypes/shapes identical for all 138,681 tensors; sampled experts byte-identical.
