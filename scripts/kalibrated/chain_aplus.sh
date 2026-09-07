#!/usr/bin/env bash
# chain_aplus.sh - A+ on the pod: wait for source + build, view, recipe (K3 on the 26 chosen layers, K2 on the other
# 17, all 256 experts), convert (mcg, ROWS rows, 2 GPUs, compile skipped), splice every finished K3 layer into
# /workspace/splice-aplus as a 256-expert file in Vis's K3 layer layout (identity plan) so the Spark can pull during the run.
set -u -o pipefail
cd /workspace; mkdir -p logs
T=/workspace/tools; SRC=/workspace/src-ablit; VIEW=/workspace/view; WORK=/workspace/work; OUT=/workspace/out-unused
RCP=/workspace/recipe_aplus.yaml; SPL=/workspace/splice-aplus; ROWS=${ROWS:-250}; CPI=${CPI:-600}
K3="0 1 6 8 9 11 12 16 19 23 24 25 26 27 29 30 31 32 33 34 35 36 37 39 40 42"
LOG=/workspace/logs/convert.log; CONVPAT='convert.py -i /workspace/view'
is_k3(){ case " $K3 " in *" $1 "*) return 0;; *) return 1;; esac; }
qt_ok(){ python - $WORK/qtensors/layers.$1.safetensors <<'PY'
import sys,os,json,struct
p=sys.argv[1]
if not os.path.exists(p): sys.exit(1)
fh=open(p,'rb'); n=struct.unpack('<Q',fh.read(8))[0]; h=json.loads(fh.read(n))
sys.exit(0 if sum(1 for k in h if k.endswith('.trellis') and '.ffn.experts.' in k)==768 else 1)
PY
}
splice_layer(){ python $T/splice_qt.py --qtensors $WORK/qtensors --b-headers $T/aplus-headers.json --plan $T/identity_plan.json --out $SPL --layers $1 > /workspace/logs/splice-$1.log 2>&1 && echo "=== $(date -u +%T) spliced layer $1" || { echo "### SPLICE FAILED layer $1: $(tail -n 2 /workspace/logs/splice-$1.log | tr '\n' ' ' | cut -c1-200)"; return 1; }; }
echo "### CHAIN A+ start $(date -Is) rows=$ROWS cpi=$CPI K3 layers: $K3"
while ! grep -q '^### DOWNLOAD exit' logs/download.log 2>/dev/null; do sleep 30; done
grep -q '^### DOWNLOAD exit 0' logs/download.log || { echo "### DOWNLOAD FAILED"; tail -n 5 logs/download.log | tr '\r' '\n' | tail -n 3; exit 2; }
while ! grep -q '^### BUILD exit' logs/build.log 2>/dev/null; do sleep 30; done
( grep -q '^### BUILD exit 0' logs/build.log && grep -q 'exllamav3 import OK' logs/build.log ) || { echo "### BUILD FAILED"; tail -n 20 logs/build.log | cut -c1-200; exit 3; }
echo "=== $(date -Is) source: $(ls $SRC/model-*.safetensors | wc -l) shards, $(du -sh $SRC | cut -f1); $(grep 'import OK' logs/build.log)"
echo "=== $(date -Is) view"
rm -rf $VIEW; python $T/make_view.py --src $SRC --out $VIEW --layers 0 || { echo "### VIEW FAILED"; exit 4; }
python - $VIEW <<'PY' || { echo "### VIEW SANITY FAILED"; exit 5; }
import json,sys; c=json.load(open(f"{sys.argv[1]}/config.json")); print("view: layers",c["num_hidden_layers"],"experts",c["n_routed_experts"]); assert c["n_routed_experts"]==256 and c["num_hidden_layers"]==43
PY
echo "=== $(date -Is) recipe"
( cd /workspace/exl3 && python $T/make_recipe.py --view $VIEW --out $RCP --expert-bits 3 ) || { echo "### RECIPE FAILED"; exit 6; }
python - $RCP "$K3" <<'PY' || { echo "### RECIPE PATCH FAILED"; exit 7; }
import yaml,sys,re,collections
p=sys.argv[1]; k3=set(int(x) for x in sys.argv[2].split()); r=yaml.safe_load(open(p))
pat=re.compile(r'(?:^|\.)layers\.(\d+)\.ffn\.experts\.\d+\.w[123]$'); n2=n3=0
for k in list(r["tensors"]):
    m=pat.search(k)
    if not m: continue
    if int(m.group(1)) in k3: n3+=1
    else: r["tensors"][k]=2; n2+=1
assert n3==26*768 and n2==17*768, (n3,n2)
yaml.safe_dump(r, open(p,"w"), default_flow_style=False, sort_keys=False)
c=collections.Counter(yaml.safe_load(open(p))["tensors"].values()); print("recipe patched:",dict(c)); assert c[3]==26*768 and c[2]==17*768, c
PY
echo "=== $(date -Is) convert: 256 experts, K3 on 26 layers + K2 on 17, mcg, $ROWS rows, devices 0,1; compile skipped"
rm -rf $WORK $OUT $SPL; mkdir -p $WORK $SPL
S=$(date +%s)
( cd /workspace/exl3 && python convert.py -i $VIEW -o $OUT -w $WORK -rcp $RCP -cb mcg -cr $ROWS -cpi $CPI -d 0,1 -v > $LOG 2>&1 ) &
sleep 20; last=-1; killed=0; nextspl=0
while pgrep -f "$CONVPAT" >/dev/null; do
  cur=$(grep -oE 'Quantized: layers\.[0-9]+\.ffn\.experts\.0\.w1 ' $LOG 2>/dev/null | tail -n 1 | grep -oE '[0-9]+' | head -n 1)
  if [ -n "${cur:-}" ] && [ "$cur" -ge "$last" ]; then
    if [ "$cur" -gt 0 ]; then el=$(( $(date +%s) - S )); per=$(( el / cur )); echo "=== $(date -u +%T) layer $cur started; $(( per / 60 )) min/layer; ETA quant end $(date -u -d "@$(( $(date +%s) + per * (43 - cur) ))" '+%a %H:%M') UTC"; fi
    last=$((cur+1))
  fi
  if [ -n "${cur:-}" ] && [ "$nextspl" -lt "$cur" ] && [ "$nextspl" -le 42 ] && qt_ok $nextspl; then
    if is_k3 $nextspl; then splice_layer $nextspl && nextspl=$((nextspl+1)); else nextspl=$((nextspl+1)); fi
    continue
  fi
  if grep -qE '^ -- (Writing into|Creating directory)' $LOG 2>/dev/null || ls $OUT/*.safetensors >/dev/null 2>&1; then
    echo "=== $(date -u +%T) quantization finished; stopping the converter before the compile"; pkill -f "$CONVPAT"; killed=1; sleep 5; break
  fi
  sleep 30
done
E=$(date +%s); echo "=== $(date -u +%T) convert done killed=$killed wall=$(( (E-S)/3600 ))h$(( ((E-S)%3600)/60 ))m"; grep -E '^ -- Quantized: layers\.[0-9]+ ' $LOG | tail -n 2 | cut -c1-150
grep -q Traceback $LOG && { echo "### CONVERT FAILED"; grep -n -A14 Traceback $LOG | tail -n 24 | cut -c1-200; exit 8; }
bad=""; for L in $K3; do qt_ok $L || bad="$bad $L"; done
[ -z "$bad" ] || { echo "### QTENSORS INCOMPLETE (K3 layers):$bad"; exit 8; }
echo "qtensors check: all 26 K3 layers complete"; rm -rf $WORK/ckpt $WORK/ckpt_old $WORK/ckpt_new
for L in $K3; do [ -f $SPL/exl3-layer-$(printf %03d $L)-tp1-rank0.safetensors.sha256 ] || splice_layer $L || exit 9; done
echo "splice: $(ls $SPL | wc -l) files, $(du -sh $SPL | cut -f1)"; df -h /workspace | tail -1
echo "### POD done $(date -Is)"
