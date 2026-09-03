#!/usr/bin/env python3
"""Write config.json + quantization_config.json for the converted dsvision tp1 dir.
Base = vcruz305 pack config (architecture truth for Vision-Exp) with the 0xSero rank-sliced
metadata blocks the sparkinfer image expects; vision keys dropped (text-only serve)."""
import json, sys, os
src_cfg, ref_cfg, out_dir, bitrates_path = sys.argv[1:5]
src = json.load(open(src_cfg)); ref = json.load(open(ref_cfg)); bits = json.load(open(bitrates_path))
cfg = {k: v for k, v in src.items() if not k.startswith("vision_")}
k_values = sorted({b for L in bits.values() for b in L["routed"]})
base_fp8 = ref["exl3_base_quantization_config"]
cfg["exl3_base_quantization_config"] = base_fp8
cfg["hybrid_tr3_tail"] = {
    "bits": "mixed",
    "k_values": k_values,
    "bits_per_expert": "bitrates.json:routed",
    "codebook": "mcg",
    "exllamav3_revision": ref["hybrid_tr3_tail"].get("exllamav3_revision", "unknown"),
    "exllamav3_version": "vcruz305-mixedk-0.0.43",
    "experts_per_layer": 256,
    "format": "exl3-trellis",
    "moe_layers": [0, 42],
    "source_operation": "convert_vision_pack (HF-sharded -> rank-sliced tp1)",
    "source_revision": "vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK",
    "source_tp": 1,
    "tensor_schema": ref["hybrid_tr3_tail"]["tensor_schema"],
    "tp": 1,
}
cfg["quantization_config"] = {
    "base_quantization_config": base_fp8,
    "bits": 2.0,
    "codebook": "mcg",
    "quant_method": "exl3",
    "source_format": "packed_e2m1_fp4_with_ue8m0_scales",
    "version": "rank-sliced-deepseek-v4-v1",
    "layer_bits": src.get("quantization_config", {}).get("layer_bits", {}),
}
cfg["n_routed_experts"] = 256
cfg["num_nextn_predict_layers"] = ref.get("num_nextn_predict_layers", 1)  # mirror the reference (3 DSpark stages regardless)
cfg["expert_dtype"] = "fp4"
cfg["torch_dtype"] = "bfloat16"
cfg["transformers_version"] = ref.get("transformers_version", "4.57.1")
cfg['rms_norm_eps'] = 1e-06  # sparkinfer fused mHC Gram kernel requires 1e-6 (pack says 1e-20, numerically irrelevant)
json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2, sort_keys=True)
qc = dict(cfg["quantization_config"]); qc.pop("layer_bits", None)
qc["rank_sliced"] = dict(cfg["hybrid_tr3_tail"]); qc["source_operation"] = cfg["hybrid_tr3_tail"]["source_operation"]
json.dump(qc, open(os.path.join(out_dir, "quantization_config.json"), "w"), indent=2, sort_keys=True)
print("config written; k_values", k_values, "K3 layers:", [L for L in bits if bits[L]["routed"][0] == 3])
