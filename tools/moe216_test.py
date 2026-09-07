#!/opt/runtime-venv/bin/python
"""A/B of sparkinfer's fused EXL3 MoE with the SAME expert weights at two expert counts.

A = Vis pack (256 slots), routed only to the REAP-kept experts.
B = pack E (216 slots) = the same expert tensors, renumbered to the kept-slot index.
Both are run through the exact serving paths (prefill plan block_m=64 cap 4096,
decode plan block_m=8 cap 32) and compared row by row.
"""
import argparse, json, struct, time, torch
from sparkinfer.moe import fused_moe as api

DEV = torch.device("cuda:0")
H, I, TOPK = 4096, 2048, 6
TILE = (64, 256, 64, 256)          # _trellis_tile_config(4096, 2048)
DT = torch.bfloat16
VIS = "/models/deepseek-v4-flash-vision-spark/tp1"
PACKE = "/models/dsvision3e-splice"
PLAN = "/models/deepseek-v4-flash-vision-k216-spark/tp1/REAP_K216_PLAN.json"


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


def load_layer(path, L):
    st = ST(path)
    E = sum(1 for k in st.h if k.startswith(f"layers.{L}.ffn.experts.") and k.endswith(".w1.rank0.trellis"))
    k = lambda e, w, s: f"layers.{L}.ffn.experts.{e}.{w}.rank0.{s}"
    stack = lambda w, s: torch.stack([st.get(k(e, w, s)) for e in range(E)])
    w1, w3, w2 = stack("w1", "trellis"), stack("w3", "trellis"), stack("w2", "trellis")
    bits = w1.shape[-1] // 16
    slabs = dict(
        w13=torch.stack([w1, w3]).to(DEV).contiguous(),
        w2=w2.to(DEV).contiguous(),
        gate_suh=stack("w1", "suh").to(DEV).contiguous(),
        up_suh=stack("w3", "suh").to(DEV).contiguous(),
        rot=torch.cat([stack("w1", "svh"), stack("w3", "svh"), stack("w2", "suh")], dim=1).to(DEV).contiguous(),
        down_svh=stack("w2", "svh").to(DEV).contiguous(),
        mcg=st.get(k(0, "w1", "mcg")).to(DEV),
    )
    assert tuple(slabs["w13"].shape) == (2, E, H // 16, I // 16, 16 * bits), slabs["w13"].shape
    assert tuple(slabs["w2"].shape) == (E, I // 16, H // 16, 16 * bits), slabs["w2"].shape
    return E, bits, slabs


def prepare(E, bits, s):
    wp = api.plan_weights(quant_modes="w4a16", source_format="exl3_trellis_mcg", activation="silu",
                          params_dtype=DT, num_experts=E, hidden_size=H, intermediate_size=I,
                          w13_layout="w13", trellis_bits=bits, trellis_tile_config=TILE)
    prepared = api.prepare_weights(plan=wp, params_dtype=DT, w1_fp4=s["w13"], w2_fp4=s["w2"],
                                   gate_suh=s["gate_suh"], up_suh=s["up_suh"], intermediate_rotations=s["rot"],
                                   down_svh=s["down_svh"], trellis_mcg=s["mcg"])
    return wp, prepared


def make_plan(wp, max_tokens, block_m):
    caps = api.Caps(max_tokens=max_tokens, num_topk=TOPK, route_num_experts=0, device=DEV,
                    weight_plan=wp, quant_mode="w4a16", w4a16_block_size_m=block_m)
    plan = api.plan(caps)
    spec = plan.scratch_specs()[0]
    return plan, torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)


def run(plan, scratch, prepared, x, w, ids):
    b = api.bind(plan, scratch=scratch, a=x, experts=prepared, topk_weights=w, topk_ids=ids)
    return api.run(binding=b).detach().clone().float()


def run_decode(plan, scratch, prepared, x, w, ids, step=32):
    return torch.cat([run(plan, scratch, prepared, x[i:i + step], w[i:i + step], ids[i:i + step])
                      for i in range(0, x.shape[0], step)])


def cmp(name, a, b, ids_slot=None, E=None):
    d = (a - b).norm(dim=1) / a.norm(dim=1).clamp_min(1e-6)
    bad = d > 0.05
    print(f"{name:34s} rel/row mean {d.mean():.5f} p50 {d.median():.5f} p95 {d.quantile(0.95):.5f} "
          f"max {d.max():.4f}  rows>5% {int(bad.sum())}/{len(d)}", flush=True)
    if ids_slot is not None and bad.any():
        blk = 512
        print("   by position:", " ".join(f"{d[i:i+blk].mean():.4f}" for i in range(0, len(d), blk)))
        allc = torch.bincount(ids_slot.flatten(), minlength=E).float()
        badc = torch.bincount(ids_slot[bad].flatten(), minlength=E).float()
        ratio = badc / allc.clamp_min(1)
        top = torch.argsort(ratio, descending=True)[:12]
        print("   slots most implicated (slot: badrows/allrows):",
              " ".join(f"{int(s)}:{int(badc[s])}/{int(allc[s])}" for s in top))
        for lo in (0, 64, 128, 192, 208):
            sel = slice(lo, E)
            print(f"   slots>={lo}: bad-rate {badc[sel].sum() / allc[sel].sum().clamp_min(1):.4f}", end="")
        print()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--out", default="")
    ap.add_argument("--skew", default="uniform", choices=["uniform", "zipf", "tail"])
    ap.add_argument("--xscale", type=float, default=1.0)
    a = ap.parse_args()
    L = a.layer
    torch.manual_seed(0)
    kept = json.load(open(PLAN))["keep_maps"]["keep_by_layer"][str(L)]
    t0 = time.time()
    EA, bitsA, sA = load_layer(f"{VIS}/exl3-layer-{L:03d}-tp1-rank0.safetensors", L)
    EB, bitsB, sB = load_layer(f"{PACKE}/exl3-layer-{L:03d}-tp1-rank0.safetensors", L)
    print(f"layer {L}: A(Vis) E={EA} K{bitsA}  B(packE) E={EB} K{bitsB}  kept={len(kept)}  load {time.time()-t0:.0f}s", flush=True)
    assert EB == len(kept) and bitsA == bitsB
    # sanity: B slot j must hold the same bytes as A expert kept[j]
    for j in (0, EB // 2, EB - 1):
        assert torch.equal(sA["w13"][0, kept[j]], sB["w13"][0, j]), f"slot {j} differs from expert {kept[j]}"
        assert torch.equal(sA["w2"][kept[j]], sB["w2"][j])
    print("   slot<->expert byte check OK", flush=True)

    M = a.m
    x = (torch.randn(M, H, device=DEV) * a.xscale).to(DT)
    if a.skew == "uniform":
        slot = torch.rand(M, EB, device=DEV).argsort(dim=1)[:, :TOPK].contiguous()      # 6 distinct kept slots
    elif a.skew == "zipf":   # real routing is skewed: expert popularity ~ 1/rank
        pop = 1.0 / (torch.arange(EB, device=DEV).float() + 1.0)
        pop = pop[torch.randperm(EB, device=DEV)]
        slot = torch.multinomial(pop.expand(M, EB), TOPK, replacement=False).contiguous()
    else:                    # everything on the last 8 slots (208..215): max rows per expert, tail slots
        slot = (EB - 8 + torch.rand(M, 8, device=DEV).argsort(dim=1)[:, :TOPK]).contiguous()
    print(f"   routing {a.skew}: rows/slot max {int(torch.bincount(slot.flatten(), minlength=EB).max())} min {int(torch.bincount(slot.flatten(), minlength=EB).min())}", flush=True)
    idsB = slot.to(torch.int64)
    idsA = torch.tensor(kept, device=DEV, dtype=torch.int64)[slot].contiguous()
    w = (torch.softmax(torch.randn(M, TOPK, device=DEV), dim=1) * 1.5).contiguous()

    t0 = time.time()
    wpA, prA = prepare(EA, bitsA, sA)
    wpB, prB = prepare(EB, bitsB, sB)
    print(f"   prepared both in {time.time()-t0:.0f}s", flush=True)
    res = {}
    for tag, wp, pr, ids in (("A256", wpA, prA, idsA), ("B216", wpB, prB, idsB)):
        t0 = time.time()
        pplan, pscr = make_plan(wp, 4096, 64)
        res[tag + "_prefill"] = run(pplan, pscr, pr, x, w, ids)
        torch.cuda.synchronize()
        t1 = time.time()
        dplan, dscr = make_plan(wp, 32, 8)
        res[tag + "_decode"] = run_decode(dplan, dscr, pr, x, w, ids)
        torch.cuda.synchronize()
        print(f"   {tag}: prefill(M={M}) {t1-t0:.1f}s  decode(32-row chunks) {time.time()-t1:.1f}s", flush=True)
        del pplan, pscr, dplan, dscr
    print()
    cmp("A256 prefill vs A256 decode", res["A256_prefill"], res["A256_decode"], slot, EB)
    cmp("B216 prefill vs B216 decode", res["B216_prefill"], res["B216_decode"], slot, EB)
    cmp("A256 vs B216  (prefill)", res["A256_prefill"], res["B216_prefill"], slot, EB)
    cmp("A256 vs B216  (decode)", res["A256_decode"], res["B216_decode"], slot, EB)
    if a.out:
        n = 256
        torch.save({"layer": L, "kept": kept, "x": x[:n].cpu(), "slot": slot[:n].cpu(), "w": w[:n].cpu(),
                    **{k: v[:n].cpu() for k, v in res.items()}}, a.out)
        print("saved", a.out)


if __name__ == "__main__":
    main()
