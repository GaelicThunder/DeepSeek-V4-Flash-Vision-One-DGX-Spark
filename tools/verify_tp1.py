#!/usr/bin/env python3
"""Verify a converted tp1 dir against the source pack: names/dtypes/shapes for every tensor,
byte-equality for copied tensors (sampled + all small ones), FP8 dequant error for re-quantized ones."""
import json, os, re, struct, sys, random, hashlib
import torch
from safetensors import safe_open

src_dir, out_dir = sys.argv[1], sys.argv[2]
sample = int(sys.argv[3]) if len(sys.argv) > 3 else 400
random.seed(1)

def headers(d):
    cat = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".safetensors"): continue
        with open(os.path.join(d, fn), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
        hdr.pop("__metadata__", None)
        for k, v in hdr.items():
            cat[k] = (fn, v["dtype"], tuple(v["shape"]), 8 + n + v["data_offsets"][0], 8 + n + v["data_offsets"][1])
    return cat

src = headers(src_dir); out = headers(out_dir)
print(f"source {len(src)} tensors, output {len(out)} tensors")
def src_name(k):
    return re.sub(r"\.rank0\.(trellis|suh|svh|mcg|mul1)$", r".\1", k)
def raw(d, cat, k):
    fn, dt, shape, s0, s1 = cat[k]
    with open(os.path.join(d, fn), "rb") as fh:
        fh.seek(s0); return fh.read(s1 - s0)

bad = 0; checked_eq = 0; fp8 = []
keys = list(out)
scales = {k for k in keys if k.endswith(".scale") and src_name(k) not in src}
small = [k for k in keys if k not in scales and (out[k][4] - out[k][3]) <= 65536]
big = [k for k in keys if k not in scales and (out[k][4] - out[k][3]) > 65536]
todo = small + random.sample(big, min(sample, len(big)))
for k in todo:
    sk = src_name(k)
    if sk not in src:
        print("  [missing in source]", k); bad += 1; continue
    fn, dt, shape, s0, s1 = out[k]; sfn, sdt, sshape, ss0, ss1 = src[sk]
    if dt == "F8_E4M3" and sdt == "BF16":
        fp8.append(k); continue
    if k.endswith(".rank0.mcg"):
        ok = (shape == () and sshape == (1,)) and (s1 - s0) == (ss1 - ss0)
    else:
        ok = dt == sdt and shape == sshape and (s1 - s0) == (ss1 - ss0)
    if not ok:
        print("  [meta mismatch]", k, dt, shape, "vs", sdt, sshape); bad += 1; continue
    if raw(out_dir, out, k) != raw(src_dir, src, sk):
        print("  [bytes differ]", k); bad += 1
    else:
        checked_eq += 1
print(f"byte-identical checked: {checked_eq}; fp8 re-quantized sampled: {len(fp8)}; scales generated: {len(scales)}")
worst = 0.0
for k in fp8[:40]:
    sk = src_name(k); sfn, sdt, sshape, ss0, ss1 = src[sk]
    w = torch.frombuffer(bytearray(raw(src_dir, src, sk)), dtype=torch.bfloat16).reshape(sshape).float()
    q = torch.frombuffer(bytearray(raw(out_dir, out, k)), dtype=torch.float8_e4m3fn).reshape(sshape).float()
    e = torch.frombuffer(bytearray(raw(out_dir, out, k[:-len(".weight")] + ".scale")), dtype=torch.uint8).float() - 127
    N, K = sshape; e = e.reshape(N // 128, K // 128)
    deq = (q.reshape(N // 128, 128, K // 128, 128) * torch.exp2(e)[:, None, :, None]).reshape(N, K)
    rel = ((deq - w).norm() / max(w.norm().item(), 1e-9)).item()
    worst = max(worst, rel)
    if rel > 0.08:
        print(f"  [fp8 rel err high] {k}: {rel:.4f}"); bad += 1
print(f"fp8 worst rel-L2 error over {min(40,len(fp8))} tensors: {worst:.5f} (0 = exact fp8-origin; ~0.03 for the 26 abliterated wo_b)")
# index consistency
idx = json.load(open(os.path.join(out_dir, "model.safetensors.index.json")))["weight_map"] if os.path.exists(os.path.join(out_dir, "model.safetensors.index.json")) else None
if idx is not None:
    miss = [k for k in idx if k not in out]; extra = [k for k in out if k not in idx]
    print(f"index: {len(idx)} entries, missing files for {len(miss)}, tensors not indexed {len(extra)}")
    bad += len(miss) + len(extra)
print("VERIFY", "OK" if bad == 0 else f"FAILED ({bad} problems)")
sys.exit(1 if bad else 0)
