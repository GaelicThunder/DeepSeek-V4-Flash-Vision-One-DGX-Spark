#!/usr/bin/env bash
# switch.sh - at the next checkpoint: stop the running chain + converter, install args.json.new (remaining non-K3
# layers at 16 bit), and launch chain_aplus_resume.sh. Bounded work loss = time since that checkpoint.
set -u; cd /workspace
J=/workspace/work/ckpt/job.json; m0=$(stat -c %Y $J)
[ -f /workspace/work/args.json.new ] || { echo "### SWITCH ABORTED: args.json.new missing"; exit 1; }
echo "=== $(date -u +%T) waiting for the next checkpoint (job.json mtime $m0, $(tr -d '\n ' < $J))"
for i in $(seq 1 150); do [ "$(stat -c %Y $J)" != "$m0" ] && break; sleep 10; done
[ "$(stat -c %Y $J)" != "$m0" ] || { echo "### SWITCH ABORTED: no new checkpoint in 25 min, nothing touched"; exit 2; }
sleep 20; while [ -d /workspace/work/ckpt_new ]; do sleep 5; done
echo "=== $(date -u +%T) new checkpoint: $(tr -d '\n ' < $J)"
pkill -f 'bash /workspace/tools/chain_aplus.sh'; sleep 2
pkill -TERM -f 'convert.py -i /workspace/view'
for i in $(seq 1 90); do pgrep -f 'convert.py -i /workspace/view' >/dev/null || break; sleep 2; done
pgrep -f 'convert.py -i /workspace/view' >/dev/null && { pkill -KILL -f 'convert.py -i /workspace/view'; sleep 5; }
pgrep -fa 'chain_aplus.sh|convert.py -i' >/dev/null && { echo "### SWITCH ABORTED: old processes still alive"; exit 3; }
echo "=== $(date -u +%T) old run stopped; ckpt job=$(tr -d '\n ' < $J) state=$(stat -c %s /workspace/work/ckpt/state.safetensors) bytes"
cp /workspace/work/args.json /workspace/work/args.json.k2 && cp /workspace/work/args.json.new /workspace/work/args.json || { echo "### SWITCH ABORTED: args.json swap failed"; exit 4; }
python - <<'PY'
import json, collections
a = json.load(open('/workspace/work/args.json')); print("args.json strategy now:", dict(collections.Counter(a['recipe_strategy'].values())), "codebook", a['codebook'], "cal_rows", a['cal_rows'])
PY
setsid nohup bash /workspace/tools/chain_aplus_resume.sh >> /workspace/logs/chain.log 2>&1 < /dev/null &
sleep 3; echo "=== $(date -u +%T) resume chain launched ($(pgrep -fc chain_aplus_resume.sh) proc)"
