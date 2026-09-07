#!/usr/bin/env bash
# pull_aplus.sh - pull the 26 A+ K3 layer files (256 experts, Vis layout) from the RunPod pod as they appear (07-09).
# NOTE: remote commands run through ssh are DOUBLE-quoted so the single-quoted grep pattern survives to the pod shell.
set -u
H=root@POD_IP; P=POD_PORT; SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $P"
K3="0 1 6 8 9 11 12 16 19 23 24 25 26 27 29 30 31 32 33 34 35 36 37 39 40 42"
DST=$HOME/models/dsvision-aplus-splice; mkdir -p $DST; cd $DST
echo "### PULL start $(date -Is)"
for L in $K3; do
  f=$(printf 'exl3-layer-%03d-tp1-rank0.safetensors' $L)
  if [ -f $DST/$f.sha256 ] && read sha bytes < $DST/$f.sha256 && [ "$(stat -c %s $DST/$f 2>/dev/null)" = "$bytes" ]; then echo "=== layer $L already here"; continue; fi
  until $SSH $H "test -f /workspace/splice-aplus/$f.sha256" 2>/dev/null; do
    if $SSH $H "grep -qE '^###.*(FAILED|INCOMPLETE|ABORTED)' /workspace/logs/chain.log" 2>/dev/null; then echo "### PULL ABORTED: chain failed"; $SSH $H "grep -E '^###.*(FAILED|INCOMPLETE|ABORTED)' /workspace/logs/chain.log" 2>/dev/null | tail -n 2; exit 2; fi
    sleep 60
  done
  S=$(date +%s)
  until rsync -a --partial -e "$SSH" $H:/workspace/splice-aplus/$f $H:/workspace/splice-aplus/$f.sha256 $DST/ 2>>$DST/pull.err; do sleep 30; done
  read sha bytes < $DST/$f.sha256; sz=$(stat -c %s $DST/$f); E=$(date +%s)
  if [ "$sz" = "$bytes" ] && [ "$(sha256sum $DST/$f | cut -d' ' -f1)" = "$sha" ]; then echo "=== $(date +%T) layer $L ok $((E-S)) s, $(( sz / 1048576 / (E-S+1) )) MB/s"; else echo "### layer $L MISMATCH"; fi
done
echo "### PULL done $(date -Is): $(ls $DST/*.safetensors 2>/dev/null | wc -l) files, $(du -sh $DST | cut -f1)"
