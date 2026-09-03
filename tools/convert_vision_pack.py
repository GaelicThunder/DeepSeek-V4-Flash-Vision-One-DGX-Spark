#!/usr/bin/env python3
"""Convert vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK (HF-sharded, BF16 non-routed) into the
0xSero "rank-sliced TP1" layout that the sparkinfer image loads:

  exl3-layer-LLL-tp1-rank0.safetensors   routed experts, byte-identical trellis payloads,
                                          names layers.L.ffn.experts.E.wP.rank0.{trellis,suh,svh,mcg}
  carried-00N.safetensors                 everything else; BF16 projections re-quantized to
                                          FP8 e4m3 + ue8m0 128x128 block scales (.scale, F8_E8M0)
                                          exactly where the reference tp1 has them
  model.safetensors.index.json, bitrates.json, rank-sliced-tp1-manifest.json, tokenizer files
  (config.json / quantization_config.json: make_config.py)

Dropped: vision.*, aligner.*, image_*, *.gate.bias_vl, hash-layer (0..2) gate.bias.
I/O: source shards are read in offset order (sequential on the USB disk); a layer's expert span is
read in one shot (~1.5-2.3 GiB RAM). Trellis payloads are never decoded.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, struct, time
from collections import OrderedDict, defaultdict
import torch

EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(trellis|suh|svh|mcg|mul1)$")
DROP_PREFIX = ("vision.", "aligner.", "image_")
FP8_TARGETS = re.compile(
    r"^(layers|mtp)\.\d+\.(attn\.(wq_a|wq_b|wkv|wo_a|wo_b|indexer\.wq_b)\.weight"
    r"|ffn\.shared_experts\.w[123]\.weight|main_proj\.weight)$"
)
BLOCK = 128
E4M3_MAX = 448.0
CHUNK = 64 << 20


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def catalog(src_dir):
    cat = {}
    for fn in sorted(os.listdir(src_dir)):
        if not fn.endswith(".safetensors"):
            continue
        hdr, base = read_header(os.path.join(src_dir, fn))
        for k, v in hdr.items():
            o0, o1 = v["data_offsets"]
            cat[k] = (fn, v["dtype"], tuple(v["shape"]), base + o0, base + o1)
    return cat


class Sha256Writer:
    def __init__(self, path):
        self.fh = open(path, "wb", buffering=CHUNK)
        self.h = hashlib.sha256()
        self.n = 0

    def write(self, b):
        self.fh.write(b)
        self.h.update(b)
        self.n += len(b)

    def close(self):
        self.fh.close()
        return self.h.hexdigest(), self.n


def read_span(path, start, end):
    with open(path, "rb") as fh:
        fh.seek(start)
        buf = bytearray(end - start)
        mv = memoryview(buf)
        got = 0
        while got < len(buf):
            n = fh.readinto(mv[got:])
            if not n:
                raise IOError(f"short read in {path}")
            got += n
    return buf


def bf16_to_fp8_block(w: torch.Tensor):
    """w [N,K] bf16 -> (q bytes F8_E4M3 [N,K], scale bytes F8_E8M0 [N/128,K/128]) with ue8m0 power-of-two scales."""
    N, K = w.shape
    assert N % BLOCK == 0 and K % BLOCK == 0, (N, K)
    x = w.to(torch.float32).reshape(N // BLOCK, BLOCK, K // BLOCK, BLOCK)
    amax = x.abs().amax(dim=(1, 3))
    exp = torch.where(amax > 0, torch.ceil(torch.log2(amax / E4M3_MAX)), torch.zeros_like(amax)).clamp(-127, 127)
    scale = torch.exp2(exp)
    q = (x / scale[:, None, :, None]).reshape(N, K).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    e8m0 = (exp + 127).to(torch.uint8)
    return q.view(torch.uint8).contiguous().numpy().tobytes(), e8m0.contiguous().numpy().tobytes()


def st_header(entries):
    hdr = OrderedDict()
    hdr["__metadata__"] = {"format": "pt"}
    off = 0
    for name, dtype, shape, nbytes in entries:
        hdr[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + nbytes]}
        off += nbytes
    js = json.dumps(hdr, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 8)
    return struct.pack("<Q", len(js)) + js, off


def load_spans(src_dir, items):
    """items: list of (src_fn, s0, s1). Read per file in offset order; one span read if the span is tight.
    Returns dict (fn,s0,s1) -> bytes-like."""
    out = {}
    byfile = defaultdict(list)
    for fn, s0, s1 in items:
        byfile[fn].append((s0, s1))
    for fn, spans in byfile.items():
        spans.sort()
        need = sum(s1 - s0 for s0, s1 in spans)
        lo, hi = spans[0][0], max(s1 for _, s1 in spans)
        path = os.path.join(src_dir, fn)
        if hi - lo <= need * 1.25 + (64 << 20):
            buf = read_span(path, lo, hi)
            mv = memoryview(buf)
            for s0, s1 in spans:
                out[(fn, s0, s1)] = mv[s0 - lo:s1 - lo]
        else:
            with open(path, "rb") as fh:
                for s0, s1 in spans:
                    fh.seek(s0)
                    out[(fn, s0, s1)] = fh.read(s1 - s0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref-spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--carried-bytes", type=int, default=4 << 30)
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--only", default="", help="debug: 'experts' or 'carried'")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    cat = catalog(args.src)
    ref = json.load(open(args.ref_spec))["spec"]
    log = open(os.path.join(args.out, "convert.log"), "a")

    def emit(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    emit(f"catalog: {len(cat)} source tensors")
    weight_map, manifest_files, bitrates = {}, [], {}
    total_bytes = total_count = 0

    def finish_file(fn, entries, sha, n):
        nonlocal total_bytes, total_count
        for name, *_ in entries:
            weight_map[name] = fn
        manifest_files.append({"name": fn, "bytes": n, "sha256": sha})
        total_bytes += n; total_count += len(entries)

    # ---------------- routed experts: per-layer files ----------------
    if args.only in ("", "experts"):
        for L in range(args.layer_start, args.layers):
            fn = f"exl3-layer-{L:03d}-tp1-rank0.safetensors"
            outp = os.path.join(args.out, fn)
            entries, plan, bits_layer = [], [], None
            for E in range(256):
                for P in ("w1", "w2", "w3"):
                    for F in ("trellis", "suh", "svh", "mcg"):
                        k = f"layers.{L}.ffn.experts.{E}.{P}.{F}"
                        if k not in cat:
                            raise SystemExit(f"missing expert tensor {k}")
                        sfn, dt, shape, s0, s1 = cat[k]
                        if F == "mcg":
                            shape = ()
                        if F == "trellis":
                            b = shape[2] // 16
                            bits_layer = b if bits_layer is None else bits_layer
                            if bits_layer != b:
                                raise SystemExit(f"layer {L}: mixed trellis widths inside layer at {k}")
                        entries.append((f"layers.{L}.ffn.experts.{E}.{P}.rank0.{F}", dt, shape, s1 - s0))
                        plan.append((sfn, s0, s1))
            bitrates[str(L)] = {"routed": [bits_layer] * 256}
            if os.path.exists(outp) and os.path.exists(outp + ".sha256"):
                sha, n = open(outp + ".sha256").read().split(); n = int(n)
                emit(f"[skip] {fn} exists ({n/2**30:.2f} GiB)")
            else:
                data = load_spans(args.src, plan)
                hdr, nbytes = st_header(entries)
                w = Sha256Writer(outp + ".tmp"); w.write(hdr)
                for key in plan:
                    w.write(data[key])
                del data
                sha, n = w.close()
                assert n == len(hdr) + nbytes
                os.replace(outp + ".tmp", outp)
                open(outp + ".sha256", "w").write(f"{sha} {n}")
                emit(f"[ok] {fn} K{bits_layer} {n/2**30:.2f} GiB  t+{time.time()-t0:.0f}s")
            finish_file(fn, entries, sha, n)

    # ---------------- carried (non-routed) tensors, streamed in source order ----------------
    if args.only in ("", "carried"):
        keep, dropped = [], []
        for k in cat:
            if EXPERT_RE.match(k):
                continue
            if k.startswith(DROP_PREFIX) or k.endswith(".gate.bias_vl"):
                dropped.append(k); continue
            m = re.match(r"^layers\.(\d+)\.ffn\.gate\.bias$", k)
            if m and int(m.group(1)) < 3:
                dropped.append(k); continue
            keep.append(k)
        # MTP experts >= 216 have no entry in the REAP-K216 reference: use expert 0 of the same stage as template
        def refspec(k):
            return ref.get(k) or ref[re.sub(r"^(mtp\.\d+\.ffn\.experts\.)\d+\.", r"\g<1>0.", k)]
        def in_ref(k):
            try: refspec(k); return True
            except KeyError: return False
        missing = [k for k in keep if not in_ref(k)]
        if missing:
            raise SystemExit(f"{len(missing)} kept tensors absent from reference spec, e.g. {missing[:5]}")
        gen_scales = {k[:-len(".weight")] + ".scale" for k in keep if FP8_TARGETS.match(k)}
        produced = set(keep) | gen_scales
        ref_missing = [k for k in ref if not re.match(r"^layers\.\d+\.ffn\.experts\.", k) and k not in produced]
        if ref_missing:
            raise SystemExit(f"reference tensors not produced: {ref_missing[:10]} (total {len(ref_missing)})")
        emit(f"carried: {len(keep)} source tensors kept, {len(dropped)} dropped, {len(gen_scales)} FP8 scales to generate")
        keep.sort(key=lambda k: (cat[k][0], cat[k][3]))  # source file, offset -> sequential reads

        def out_size(k):
            sfn, dt, shape, s0, s1 = cat[k]
            return (shape[0] * shape[1] + (shape[0] // BLOCK) * (shape[1] // BLOCK)) if FP8_TARGETS.match(k) else s1 - s0

        groups, cur, cur_bytes = [], [], 0
        for k in keep:
            nb = out_size(k)
            if cur and cur_bytes + nb > args.carried_bytes:
                groups.append(cur); cur, cur_bytes = [], 0
            cur.append(k); cur_bytes += nb
        if cur:
            groups.append(cur)
        for gi, names in enumerate(groups, 1):
            fn = f"carried-{gi:03d}.safetensors"
            outp = os.path.join(args.out, fn)
            entries, plan = [], []
            for k in names:
                sfn, dt, shape, s0, s1 = cat[k]
                r = refspec(k)
                if FP8_TARGETS.match(k):
                    assert r["dtype"] == "F8_E4M3", (k, r)
                    sname = k[:-len(".weight")] + ".scale"; rs = ref[sname]
                    assert tuple(rs["shape"]) == (shape[0] // BLOCK, shape[1] // BLOCK), (k, rs["shape"], shape)
                    entries.append((k, "F8_E4M3", shape, shape[0] * shape[1])); plan.append(("fp8w", k))
                    entries.append((sname, "F8_E8M0", tuple(rs["shape"]), rs["shape"][0] * rs["shape"][1])); plan.append(("fp8s", k))
                else:
                    if r["dtype"] != dt or tuple(r["shape"]) != shape and not k.endswith("gate.weight") and not k.endswith("gate.bias"):
                        raise SystemExit(f"dtype/shape mismatch not handled: {k} ref {r} src {dt} {shape}")
                    entries.append((k, dt, shape, s1 - s0)); plan.append(("copy", k))
            if os.path.exists(outp) and os.path.exists(outp + ".sha256"):
                sha, n = open(outp + ".sha256").read().split(); n = int(n)
                emit(f"[skip] {fn} exists ({n/2**30:.2f} GiB)")
            else:
                hdr, nbytes = st_header(entries)
                w = Sha256Writer(outp + ".tmp"); w.write(hdr)
                pending = {}
                cur_fh, cur_fn = None, None
                for kind, k in plan:
                    sfn, dt, shape, s0, s1 = cat[k]
                    if kind == "fp8s":
                        w.write(pending.pop(k)); continue
                    if cur_fn != sfn:
                        if cur_fh: cur_fh.close()
                        cur_fh = open(os.path.join(args.src, sfn), "rb"); cur_fn = sfn
                    cur_fh.seek(s0)
                    if kind == "copy":
                        remaining = s1 - s0
                        while remaining:
                            b = cur_fh.read(min(CHUNK, remaining))
                            if not b: raise IOError(f"short read {k}")
                            w.write(b); remaining -= len(b)
                    else:  # fp8w
                        buf = cur_fh.read(s1 - s0)
                        t = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(shape)
                        qb, sb = bf16_to_fp8_block(t)
                        w.write(qb); pending[k] = sb
                if cur_fh: cur_fh.close()
                assert not pending, list(pending)[:3]
                sha, n = w.close()
                assert n == len(hdr) + nbytes, (n, len(hdr) + nbytes)
                os.replace(outp + ".tmp", outp)
                open(outp + ".sha256", "w").write(f"{sha} {n}")
                emit(f"[ok] {fn} {len(entries)} tensors {n/2**30:.2f} GiB  t+{time.time()-t0:.0f}s")
            finish_file(fn, entries, sha, n)

    # ---------------- metadata ----------------
    if args.only == "":
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
                  open(os.path.join(args.out, "model.safetensors.index.json"), "w"))
        json.dump(bitrates, open(os.path.join(args.out, "bitrates.json"), "w"))
        json.dump({"format": "rank-sliced-exl3-tp1-v1", "source": os.path.abspath(args.src), "source_tp": 1,
                   "target_tp": 1, "files": manifest_files, "tensor_bytes": total_bytes, "tensor_count": total_count},
                  open(os.path.join(args.out, "rank-sliced-tp1-manifest.json"), "w"), indent=1)
        for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            if os.path.exists(os.path.join(args.src, f)):
                open(os.path.join(args.out, f), "wb").write(open(os.path.join(args.src, f), "rb").read())
        k3 = [L for L in sorted(bitrates, key=int) if bitrates[L]["routed"][0] != 2]
        emit(f"DONE {total_count} tensors {total_bytes/2**30:.2f} GiB in {time.time()-t0:.0f}s; K3 layers: {k3} (others K2)")


if __name__ == "__main__":
    main()
