#!/usr/bin/env bash
# ramwatch.sh CONTAINER [FLOOR_MB] [INTERVAL_S] — stop the engine before unified memory runs out.
#
# On a GB10 the driver fails allocations under ~1 GB of MemAvailable and the box can hard-lock instead of OOM-ing;
# killing the container first turns that into a restart. Three consecutive samples under the floor trigger it, so a
# one-sample dip does not. Exits when the container is gone. Started by start.sh (RAMWATCH=0 disables it).
set -u
C="${1:?container name}"; FLOOR="${2:-700}"; EVERY="${3:-5}"; n=0
log(){ printf '%s ramwatch %s\n' "$(date -Is)" "$*"; }
log "watching $C: MemAvailable floor ${FLOOR} MB, every ${EVERY} s"
while :; do
  docker ps --format '{{.Names}}' | grep -qx "$C" || { log "$C is not running; exiting"; exit 0; }
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -lt "$FLOOR" ]; then n=$((n+1)); log "MemAvailable ${avail} MB < ${FLOOR} MB (${n}/3)"; else n=0; fi
  if [ "$n" -ge 3 ]; then
    log "stopping $C: MemAvailable ${avail} MB for $((3*EVERY)) s"
    docker kill "$C" >/dev/null 2>&1; docker logs --tail 5 "$C" 2>&1 | sed 's/^/   /'
    exit 3
  fi
  sleep "$EVERY"
done
