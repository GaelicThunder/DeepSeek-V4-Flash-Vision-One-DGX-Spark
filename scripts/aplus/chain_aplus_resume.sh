#!/usr/bin/env bash
# chain_aplus_resume.sh - resume the A+ conversion from its checkpoint with the remaining non-K3 layers stored at
# 16 bit (exact activations propagated, no quantization time), same monitoring and incremental splice of K3 layers.
# Launched by switch.sh after args.json has been patched. Appends to chain.log so the watchers keep following.
set -u -o pipefail
cd /workspace; T=/workspace/tools; WORK=/workspace/work; OUT=/workspace/out-unused; SPL=/workspace/splice-aplus
K3="0 1 6 8 9 11 12 16 19 23 24 25 26 27 29 30 31 32 33 34 35 36 37 39 40 42"
LOG=/workspace/logs/convert.log; CONVPAT='convert.py -w /workspace/work'
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
echo "### CHAIN A+ RESUME start $(date -Is): resuming from $(tr -d '\n ' < $WORK/ckpt/job.json); remaining non-K3 layers stored at 16 bit"
S=$(date +%s)
( cd /workspace/exl3 && python convert.py -w $WORK -r -rcp /workspace/recipe_aplus.yaml -cb mcg -cr 250 -cpi 600 -d 0,1 -v >> $LOG 2>&1 ) &
sleep 30; last=-1; killed=0; nextspl=2
while pgrep -f "$CONVPAT" >/dev/null; do
  cur=$(grep -oE 'Quantized: layers\.[0-9]+\.ffn\.experts\.0\.w1 ' $LOG 2>/dev/null | tail -n 1 | grep -oE '[0-9]+' | head -n 1)
  if [ -n "${cur:-}" ] && [ "$cur" -ge "$last" ]; then
    echo "=== $(date -u +%T) layer $cur started (resume run $(( ($(date +%s) - S) / 60 )) min)"
    last=$((cur+1))
  fi
  if [ -n "${cur:-}" ] && [ "$nextspl" -lt "$cur" ] && [ "$nextspl" -le 42 ]; then
    if is_k3 $nextspl; then
      if qt_ok $nextspl; then splice_layer $nextspl && nextspl=$((nextspl+1)); continue; fi
    else nextspl=$((nextspl+1)); continue; fi
  fi
  if grep -qE '^ -- (Writing into|Creating directory)' $LOG 2>/dev/null || ls $OUT/*.safetensors >/dev/null 2>&1; then
    echo "=== $(date -u +%T) quantization finished; stopping the converter before the compile"; pkill -f "$CONVPAT"; killed=1; sleep 5; break
  fi
  sleep 30
done
E=$(date +%s); echo "=== $(date -u +%T) convert done killed=$killed resume-wall=$(( (E-S)/3600 ))h$(( ((E-S)%3600)/60 ))m"; grep -E '^ -- Quantized: layers\.[0-9]+ ' $LOG | tail -n 2 | cut -c1-150
tail -n 400 $LOG | grep -q Traceback && { echo "### CONVERT FAILED"; tail -n 400 $LOG | grep -n -A14 Traceback | tail -n 24 | cut -c1-200; exit 8; }
bad=""; for L in $K3; do qt_ok $L || bad="$bad $L"; done
[ -z "$bad" ] || { echo "### QTENSORS INCOMPLETE (K3 layers):$bad"; exit 8; }
echo "qtensors check: all 26 K3 layers complete"; rm -rf $WORK/ckpt $WORK/ckpt_old $WORK/ckpt_new
for L in $K3; do [ -f $SPL/exl3-layer-$(printf %03d $L)-tp1-rank0.safetensors.sha256 ] || splice_layer $L || exit 9; done
echo "splice: $(ls $SPL | wc -l) files, $(du -sh $SPL | cut -f1)"; df -h /workspace | tail -1
echo "### POD done $(date -Is)"
