#!/usr/bin/env bash
# stop.sh — remove the engine container and stop the helpers start.sh launched (bad_words proxy, ramwatch).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; RUN_DIR="${RUN_DIR:-$REPO/.run}"
docker rm -f "${CONTAINER:-dsvision-spark}" >/dev/null 2>&1 && echo "stopped ${CONTAINER:-dsvision-spark}" || echo "engine not running"
for f in "$RUN_DIR"/*.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f"); name=$(basename "$f" .pid)
  if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null && echo "stopped $name (pid $pid)"; fi
  rm -f "$f"
done
