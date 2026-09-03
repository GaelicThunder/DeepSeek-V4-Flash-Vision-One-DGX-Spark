#!/usr/bin/env python3
"""Dump the reference (0xSero tp1) non-expert tensor spec: name -> dtype, shape, file. Header-only reads."""
import json, struct, sys, re, os
tp1 = sys.argv[1]; out = sys.argv[2]
idx = json.load(open(os.path.join(tp1, "model.safetensors.index.json")))["weight_map"]
files = sorted(set(idx.values()))
spec = {}
expert_re = re.compile(r"^(layers|mtp)\.\d+\.ffn\.experts\.(\d+)\.")
per_file_count = {}
for f in files:
    p = os.path.join(tp1, f)
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for k, v in hdr.items():
        if k == "__metadata__": continue
        m = expert_re.match(k)
        if m and int(m.group(2)) > 0 and k.startswith("layers."):
            continue  # keep only expert 0 of main layers as representative
        spec[k] = {"dtype": v["dtype"], "shape": v["shape"], "file": f,
                   "bytes": v["data_offsets"][1] - v["data_offsets"][0]}
    per_file_count[f] = len(hdr) - ("__metadata__" in hdr)
json.dump({"spec": spec, "per_file_count": per_file_count}, open(out, "w"))
print(len(spec), "spec entries;", sum(per_file_count.values()), "tensors total")
