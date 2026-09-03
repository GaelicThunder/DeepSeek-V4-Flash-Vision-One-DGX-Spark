#!/usr/bin/env python3
"""Append the vision-side tensors I wrongly dropped back into the tp1 checkpoint.

Adds, verbatim (BF16/F32 as in the pack): vision.* / aligner.* / image_* (tower + projector),
every `*.gate.bias_vl` (image-token routing bias, required by vLLM PR 54566) and the
hash-layer `layers.{0,1,2}.ffn.gate.bias` the vision checkpoint ships. Writes them into new
`carried-vis-NNN.safetensors` files and updates the index + manifest in place.
"""
import glob, json, os, re, struct, sys, hashlib
from safetensors import safe_open

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/dsv4-vision-ablit-exl3-mixedk")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/deepseek-v4-flash-vision-spark/tp1")
LIMIT = 4 * 2**30
WANT = re.compile(r"(^vision\.)|(^aligner\.)|(^image_)|(\.gate\.bias_vl$)|(^layers\.[012]\.ffn\.gate\.bias$)")
DT = {"BF16": 2, "F32": 4, "F16": 2, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "I32": 4, "I64": 8, "BOOL": 1}

def hdr(entries):
    off, meta = 0, {}
    for name, dt, shape, nb in entries:
        meta[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nb]}
        off += nb
    blob = json.dumps(meta, separators=(",", ":")).encode()
    pad = (-len(blob)) % 8
    return struct.pack("<Q", len(blob) + pad) + blob + b" " * pad, off

def main():
    cat = {}
    for f in sorted(glob.glob(SRC + "/model-*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        base = 8 + n
        for k, v in h.items():
            if k == "__metadata__" or not WANT.search(k):
                continue
            s0, s1 = v["data_offsets"]
            cat[k] = (f, v["dtype"], tuple(v["shape"]), base + s0, base + s1)
    if not cat:
        sys.exit("no vision tensors found in the source pack")
    idx_path = f"{OUT}/" + (os.environ.get("IDX") or "model.safetensors.index.json")
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]
    todo = sorted([k for k in cat if k not in wm], key=lambda k: (cat[k][0], cat[k][3]))
    print(f"{len(cat)} vision-side tensors in pack, {len(todo)} to add "
          f"({sum(cat[k][4]-cat[k][3] for k in todo)/2**20:.1f} MiB)")
    if not todo:
        print("nothing to do"); return
    groups, cur, sz = [], [], 0
    for k in todo:
        nb = cat[k][4] - cat[k][3]
        if cur and sz + nb > LIMIT:
            groups.append(cur); cur, sz = [], 0
        cur.append(k); sz += nb
    if cur:
        groups.append(cur)
    added, total_new = {}, 0
    for gi, names in enumerate(groups, 1):
        fn = f"carried-vis-{int(os.environ.get('GI0',0))+gi:03d}.safetensors"
        p = f"{OUT}/{fn}"
        entries = [(k, cat[k][1], cat[k][2], cat[k][4] - cat[k][3]) for k in names]
        head, nbytes = hdr(entries)
        h = hashlib.sha256(); h.update(head); written = len(head)
        with open(p + ".tmp", "wb") as w:
            w.write(head)
            fh, fhn = None, None
            for k in names:
                sfn, dt, shape, s0, s1 = cat[k]
                exp = 1
                for d in shape:
                    exp *= d
                exp *= DT[dt]
                assert exp == s1 - s0, (k, dt, shape, s1 - s0)
                if fhn != sfn:
                    if fh: fh.close()
                    fh, fhn = open(sfn, "rb", buffering=1 << 20), sfn
                fh.seek(s0)
                left = s1 - s0
                while left:
                    b = fh.read(min(left, 1 << 22))
                    if not b: raise IOError(f"short read {k}")
                    w.write(b); h.update(b); written += len(b); left -= len(b)
            if fh: fh.close()
        assert written == len(head) + nbytes, (written, len(head) + nbytes)
        os.replace(p + ".tmp", p)
        open(p + ".sha256", "w").write(f"{h.hexdigest()} {written}")
        for k in names:
            added[k] = fn
        total_new += written
        print(f"[ok] {fn} {len(names)} tensors {written/2**20:.1f} MiB")
    wm.update(added)
    idx["metadata"]["total_size"] = int(idx["metadata"].get("total_size", 0)) + total_new
    json.dump(idx, open(idx_path, "w"), indent=1)
    mp = f"{OUT}/" + (os.environ.get("MAN") or "rank-sliced-tp1-manifest.json")
    if os.path.exists(mp):
        man = json.load(open(mp))
        key = "files" if "files" in man else None
        if key:
            have = {e.get("name") for e in man[key]}
            for gi in range(1, len(groups) + 1):
                fn = f"carried-vis-{int(os.environ.get('GI0',0))+gi:03d}.safetensors"
                if fn in have: continue
                sha, n = open(f"{OUT}/{fn}.sha256").read().split()
                man[key].append({"name": fn, "bytes": int(n), "sha256": sha})
            json.dump(man, open(mp, "w"), indent=1)
            print(f"manifest updated ({len(man[key])} files)")
    # verify readable
    for gi in range(1, len(groups) + 1):
        with safe_open(f"{OUT}/carried-vis-{int(os.environ.get('GI0',0))+gi:03d}.safetensors", "pt") as h2:
            ks = list(h2.keys()); t = h2.get_tensor(ks[0])
        print(f"verify carried-vis-{gi:03d}: {len(ks)} tensors, {ks[0]} {tuple(t.shape)} {t.dtype}")
    print(f"index now {len(wm)} tensors, total {idx['metadata']['total_size']/2**30:.2f} GiB")

main()
