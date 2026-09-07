#!/usr/bin/env bash
# boot_aplus.sh - A+ = Vis (256 experts, K2 + 6 K3 layers) with K3 on the top-N ranked layers pulled from the pod
# (07-09). Waits for pull_aplus.sh, builds configs/bitrates/manifest for the mounted set, boots `yz start dsvision`
# with the files mounted over Vis's, falls back to fewer K3 layers if the boot does not fit, runs the battery.
set -u -o pipefail
cd ~/llm/dsvision3/ab
SPL=$HOME/models/dsvision-aplus-splice; V=$HOME/models/deepseek-v4-flash-vision-spark/tp1
URL=http://127.0.0.1:30021; MODEL=deepseek-v4-flash-vision-exp; BP=~/llm/dsvision3/bench-p7; O=$BP/out
RANK="27 23 31 35 32 34 37 40 1 39 25 29 8 30 19 26 24 11 12 9 0 42 33 36 16 6"
wd_on(){ systemctl --user start YOUR-WATCHDOG.service; }
while ! grep -qE '^### PULL (done|ABORTED)' pull_aplus.log 2>/dev/null; do sleep 60; done
grep -q '^### PULL ABORTED' pull_aplus.log && { echo "### BOOT ABORTED: pull aborted"; exit 2; }
grep -q MISMATCH pull_aplus.log && { echo "### BOOT ABORTED: pull mismatches"; grep MISMATCH pull_aplus.log; exit 2; }
n=$(ls $SPL/exl3-layer-*-tp1-rank0.safetensors | wc -l); bad=$(find $SPL -name 'exl3-layer-*-tp1-rank0.safetensors' ! -size 2425697800c ! -size 2425700872c | wc -l)
[ "$n" = 26 ] && [ "$bad" = 0 ] || { echo "### BOOT ABORTED: $n files, $bad with a wrong size"; exit 3; }
echo "### BOOT A+ start $(date -Is): 26 K3 files ok"
systemctl --user stop YOUR-WATCHDOG.service 2>/dev/null
try_boot(){
  N=$1; C=$SPL/cfg$N; rm -rf $C; mkdir -p $C/spl; LAYERS=$(echo $RANK | tr ' ' '\n' | head -n $N | tr '\n' ' ')
  echo "=== $(date +%T) trying $N K3 layers: $LAYERS"
  python3 - "$V" "$C" "$LAYERS" <<'PY' || return 4
import json, sys, os
V, C, layers = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split()]
for name in ("config.json", "config.text.json", "config-vision.json"):
    c = json.load(open(os.path.join(V, name)))
    lb = {str(k): int(v) for k, v in c["quantization_config"].get("layer_bits", {}).items()}
    for L in layers: lb[str(L)] = 3
    c["quantization_config"]["layer_bits"] = {k: lb[k] for k in sorted(lb, key=int)}
    c["hybrid_tr3_tail"]["exllamav3_version"] = f"aplus-k3x{len(lb)}-20260907"
    json.dump(c, open(os.path.join(C, name), "w"), indent=2)
b = json.load(open(os.path.join(V, "bitrates.json")))
for L in layers: b[str(L)] = {"routed": [3] * 256}
json.dump(b, open(os.path.join(C, "bitrates.json"), "w"))
k3 = sorted(int(k) for k, v in b.items() if set(v["routed"]) == {3}); print("configs:", len(k3), "K3 layers:", k3)
PY
  for L in $LAYERS; do f=$(printf "exl3-layer-%03d-tp1-rank0.safetensors" $L); cp $SPL/$f.sha256 $C/spl/; done
  ~/llm/venv/bin/python make_manifest.py --b "$V" --spl "$C/spl" --out "$C/rank-sliced-tp1-manifest.json" || return 4
  M="$C/rank-sliced-tp1-manifest.json:/models/tp1/rank-sliced-tp1-manifest.json:ro"
  for L in $LAYERS; do f=$(printf "exl3-layer-%03d-tp1-rank0.safetensors" $L); M="$M;$SPL/$f:/models/tp1/$f:ro;$SPL/$f.sha256:/models/tp1/$f.sha256:ro"; done
  for f in config.json config.text.json config-vision.json bitrates.json; do M="$M;$C/$f:/models/tp1/$f:ro"; done
  ~/llm/yz stop dsvision >/dev/null 2>&1 || true; docker rm -f dsvision-spark >/dev/null 2>&1 || true
  rm -f .preboot_force; bash preboot.sh 0.88 || { echo "### BOOT REFUSED by preboot"; return 10; }
  ( [ -f .preboot_force ] && export YZ_FORCE=1; export DSFLASH_UTIL=0.925 DSFLASH_EXTRA_MOUNTS="$M" DSFLASH_EXTRA_ENV="VERIFY_MODEL_CHECKSUMS=0"; ~/llm/yz start dsvision > boot-aplus$N.log 2>&1 )
  for i in $(seq 1 240); do curl -fsS --max-time 3 $URL/v1/models >/dev/null 2>&1 && return 0; docker ps --format '{{.Names}}' | grep -q '^dsvision-spark$' || break; sleep 10; done
  echo "### BOOT FAILED with $N K3 layers"; tail -c 600 boot-aplus$N.log; docker logs dsvision-spark 2>&1 | grep -iE 'error|out of memory|oom' | tail -n 4 | cut -c1-200; return 10
}
OKN=""; for N in 22 21 20; do if try_boot $N; then OKN=$N; break; fi; done
[ -n "$OKN" ] || { echo "### BOOT ABORTED: no configuration booted"; wd_on; exit 10; }
TAG=aplus$OKN-$(date +%Y%m%d); echo "=== $(date +%T) A+ up with $OKN K3 layers (tag $TAG); battery"
python3 $BP/nll_probe.py $BP/nll_corpus_60k.jsonl $O/nll-$TAG.json --base $URL --model $MODEL 2>&1 | tail -n 3
echo "--- A+ vs Vis (negative = A+ better; goal: all three domains negative)"; python3 $BP/nll_probe.py --compare $O/nll-dsvision-20260905.json $O/nll-$TAG.json 2>&1 | tail -n 6
echo "--- A+ vs source reference (positive = worse than source)"; python3 $BP/nll_probe.py --compare $O/nll-ref-source-20260906.json $O/nll-$TAG.json 2>&1 | tail -n 6
python3 $BP/ppl_probe.py $URL $MODEL $O/ppl-$TAG.json 2>&1 | tail -n 2
( cd $BP && python3 bench/bench_all.py $O/bench-$TAG.json --base $URL --only mc ) 2>&1 | tail -n 3
( cd $BP && DSBENCH_LOG=$O/dsbench-$TAG.jsonl python3 dsbench.py $TAG --model $MODEL --url $URL ) 2>&1 | grep -E "med="
echo "### BATTERY done $(date -Is)"; wd_on; systemctl --user is-active YOUR-WATCHDOG.service yzh.timer | tr '\n' ' '; echo
