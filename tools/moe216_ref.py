#!/usr/bin/env python3
# needs exllamav3 importable (the converter venv): dequantizes experts with LinearEXL3 for a float reference
"""Float reference for moe216_test.py outputs: dequantize the routed experts with exllamav3
(host venv) and recompute the MoE for the saved rows; compare with every kernel path."""
import argparse, json, os, struct, time, torch
from exllamav3.modules.quant.exl3 import LinearEXL3

DEV = torch.device("cuda")
VIS = os.path.expanduser("~/models/deepseek-v4-flash-vision-spark/tp1")


class ST:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h = json.loads(self.f.read(n))
        self.h.pop("__metadata__", None)
        self.base = 8 + n

    def get(self, key):
        m = self.h[key]
        a, b = m["data_offsets"]
        self.f.seek(self.base + a)
        buf = bytearray(self.f.read(b - a))
        dt = {"I16": torch.int16, "F16": torch.float16, "I32": torch.int32, "BF16": torch.bfloat16}[m["dtype"]]
        t = torch.frombuffer(buf, dtype=dt)
        return t.reshape(m["shape"]) if m["shape"] else t.reshape(())


def weight(st, key):
    tr = st.get(key + ".trellis").to(DEV)
    suh = st.get(key + ".suh").to(DEV)
    svh = st.get(key + ".svh").to(DEV)
    mcg = st.get(key + ".mcg").to(DEV) if key + ".mcg" in st.h else None
    mul1 = st.get(key + ".mul1").to(DEV) if key + ".mul1" in st.h else None
    lin = LinearEXL3(None, tr.shape[0] * 16, tr.shape[1] * 16, suh=suh, svh=svh, trellis=tr, mcg=mcg, mul1=mul1)
    return lin.get_weight_tensor().float()  # [in, out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt")
    a = ap.parse_args()
    d = torch.load(a.pt)
    L, kept = d["layer"], d["kept"]
    x = d["x"].to(DEV).float()
    slot = d["slot"].to(DEV)
    w = d["w"].to(DEV).float()
    n = x.shape[0]
    st = ST(f"{VIS}/exl3-layer-{L:03d}-tp1-rank0.safetensors")
    ref = torch.zeros(n, x.shape[1], device=DEV)
    used = sorted(set(slot.flatten().tolist()))
    t0 = time.time()
    for j in used:
        e = kept[j]
        k = lambda p: f"layers.{L}.ffn.experts.{e}.{p}.rank0"
        W1, W3, W2 = weight(st, k("w1")), weight(st, k("w3")), weight(st, k("w2"))
        rows, pos = torch.nonzero(slot == j, as_tuple=True)
        xi = x[rows]
        h = torch.nn.functional.silu(xi @ W1) * (xi @ W3)
        ref.index_add_(0, rows, (h @ W2) * w[rows, pos][:, None])
    print(f"reference: {len(used)} experts, {n} rows, {time.time()-t0:.0f}s")
    for key in ("A256_prefill", "A256_decode", "B216_prefill", "B216_decode"):
        y = d[key].to(DEV).float()
        r = (y - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-6)
        print(f"{key:14s} vs float ref: rel/row mean {r.mean():.5f} p50 {r.median():.5f} p95 {r.quantile(0.95):.5f} max {r.max():.4f} rows>5% {int((r>0.05).sum())}/{n}")


if __name__ == "__main__":
    main()
