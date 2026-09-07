#!/usr/bin/env bash
# make_aplus_pack.sh N - permanent A+ pack: Vis (hardlinks) + the top-N promoted K3 layer files + cfgN configs/manifest.
set -u -o pipefail
N=${1:?layers}; VROOT=$HOME/models/deepseek-v4-flash-vision-spark; V=$VROOT/tp1; SPL=$HOME/models/dsvision-aplus-splice
AROOT=$HOME/models/deepseek-v4-flash-vision-aplus-spark; A=$AROOT/tp1; C=$SPL/cfg$N
RANK="27 23 31 35 32 34 37 40 1 39 25 29 8 30 19 26 24 11 12 9 0 42 33 36 16 6"
LAYERS=$(echo $RANK | tr ' ' '\n' | head -n $N | tr '\n' ' ')
for f in rank-sliced-tp1-manifest.json bitrates.json config.json config.text.json config-vision.json; do [ -f $C/$f ] || { echo "missing $C/$f"; exit 2; }; done
[ -e $AROOT ] && { echo "$AROOT exists; remove it first"; exit 3; }
mkdir -p $A
cp -al $V/. $A/
for d in dspark-draft-k64 files hf-empty; do [ -e $VROOT/$d ] && cp -al $VROOT/$d $AROOT/$d; done
mkdir -p $AROOT/cache && cp -a $VROOT/cache/. $AROOT/cache/ 2>/dev/null || true
for L in $LAYERS; do f=$(printf "exl3-layer-%03d-tp1-rank0.safetensors" $L); rm -f $A/$f $A/$f.sha256; ln $SPL/$f $A/$f; ln $SPL/$f.sha256 $A/$f.sha256; done
for f in config.json config.text.json config-vision.json bitrates.json rank-sliced-tp1-manifest.json; do rm -f $A/$f; cp $C/$f $A/$f; done
# the text/vision view manifests must carry the promoted file sizes too (the hardlinked ones are MixedK's)
python3 - "$A" <<'PY2'
import json, sys
A = sys.argv[1]; cur = json.load(open(f"{A}/rank-sliced-tp1-manifest.json"))
text = dict(cur); text["files"] = [f for f in cur["files"] if not f["name"].startswith("carried-vis")]
json.dump(text, open(f"{A}/rank-sliced-tp1-manifest.text.json", "w"), indent=2); json.dump(cur, open(f"{A}/rank-sliced-tp1-manifest.vision.json", "w"), indent=2)
PY2
python3 - "$A" <<'PY'
import json, sys, os
A = sys.argv[1]
c = json.load(open(f"{A}/config.json")); lb = c["quantization_config"]["layer_bits"]
k3 = sorted(int(k) for k, v in lb.items() if int(v) == 3); print("config layer_bits: K3 on", len(k3), "layers:", k3)
b = json.load(open(f"{A}/bitrates.json")); bk3 = sorted(int(k) for k, v in b.items() if set(v["routed"]) == {3}); assert bk3 == k3, (bk3, k3)
m = json.load(open(f"{A}/rank-sliced-tp1-manifest.json"))
bad = [it["name"] for it in m["files"] if os.path.getsize(f"{A}/{it['name']}") != int(it["bytes"])]
print("manifest:", len(m["files"]), "files, size mismatches:", bad or "none")
PY
echo "A+ pack at $AROOT: $(ls $A | wc -l) entries in tp1; real $(du -sh $AROOT | cut -f1), apparent $(du -sh --apparent-size $AROOT | cut -f1); cache $(du -sh $AROOT/cache | cut -f1)"
