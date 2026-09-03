#!/usr/bin/env python3
"""Write <tp1>/config-vision.json: the text-view config with the vision architecture and the
vision_* hyper-parameters copied back from the source pack. Usage: make_vision_config.py PACK/config.json TP1_DIR"""
import json, sys
pack_cfg, tp1 = sys.argv[1], sys.argv[2]
src = json.load(open(pack_cfg))
text = json.load(open(f"{tp1}/config.text.json"))
cfg = dict(text)
cfg["architectures"] = ["DeepseekV4ForConditionalGeneration"]
vis = {k: v for k, v in src.items() if k.startswith("vision_")}
if not vis:
    sys.exit("the pack config has no vision_* keys; is this the Vision-Exp pack?")
cfg.update(vis)
json.dump(cfg, open(f"{tp1}/config-vision.json", "w"), indent=2, sort_keys=True)
print("config-vision.json written:", cfg["architectures"][0], sorted(vis))
