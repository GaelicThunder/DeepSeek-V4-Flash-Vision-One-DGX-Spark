#!/usr/bin/env bash
# start.sh — one command from a fresh DGX Spark to a serving endpoint.
#
#   ./start.sh                  # Kalibrated Vision Exp: download the ready-to-serve pack (~106 GB) -> serve -> proxy -> wait -> test request
#   PACK=mixedk ./start.sh      # the MixedK route: download vcruz305's pack (~95 GB), convert on the Spark (~30 min), serve
#   ./start.sh --no-wait        # return right after `docker run` (the proxy is up at once and answers 502 until the engine is)
#
# Tunables (environment):
#   KALIBRATED_REPO  Hub repo of the ready-to-serve pack (tp1/ + dspark-draft-k64/)
#                    default GaelicThunder/DeepSeek-V4-Flash-Vision-Exp-ablit-EXL3-Kalibrated
#   MODELS_DIR   where the served pack lives   default ~/models/deepseek-v4-flash-vision-kalibrated-spark (MixedK: ...-vision-spark)
#   PACK_DIR     where vcruz305's pack is downloaded (MixedK route)   default ~/models/dsv4-vision-ablit-exl3-mixedk
#   CTX          max_model_len                 default 245760
#   UTIL         gpu_memory_utilization        default 0.925 (Kalibrated) / 0.88 (MixedK); never above 0.93 on a 128 GB Spark
#   PORT         public port = the bad_words proxy (`)Skip` mask on every request)   default 30021
#   ENGINE_PORT  the engine's own port behind the proxy                              default 30031
#   PROXY=0      no proxy: the engine listens on PORT itself (then send bad_words yourself, see README)
#   RAMWATCH=0   do not start scripts/ramwatch.sh;  RAMWATCH_FLOOR_MB  its MemAvailable floor, default 700
#   HF_TOKEN     not needed: neither pack is gated
# Everything else (mounts, overlay, engine flags) is scripts/serve.sh, unchanged. Helpers' pids and logs: .run/
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KALIBRATED_REPO="${KALIBRATED_REPO:-GaelicThunder/DeepSeek-V4-Flash-Vision-Exp-ablit-EXL3-Kalibrated}"
PACK="${PACK:-kalibrated}"
WAIT=1; [ "${1:-}" = "--no-wait" ] && WAIT=0
PORT="${PORT:-30021}"; ENGINE_PORT="${ENGINE_PORT:-30031}"; CTX="${CTX:-245760}"
PROXY="${PROXY:-1}"; RAMWATCH="${RAMWATCH:-1}"; RAMWATCH_FLOOR_MB="${RAMWATCH_FLOOR_MB:-700}"
CONTAINER="${CONTAINER:-dsvision-spark}"
RUN_DIR="${RUN_DIR:-$REPO/.run}"; mkdir -p "$RUN_DIR"

say(){ printf '\033[36m==\033[0m %s\n' "$*"; }
die(){ printf '\033[31mxx\033[0m %s\n' "$*" >&2; exit 2; }

# ---- preflight ----------------------------------------------------------------------------------
command -v docker >/dev/null || die "docker is required"
docker info >/dev/null 2>&1 || die "docker is installed but not usable by this user (docker group? daemon running?)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/   gpu: /' || echo "   (nvidia-smi not found; continuing)"
command -v hf >/dev/null || die "the Hub CLI is required: pip install -U 'huggingface_hub[hf_xet]'   (hf_xet makes the download several times faster)"
command -v python3 >/dev/null || die "python3 is required (the bad_words proxy is standard-library Python)"
mem_gb=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo); [ "$mem_gb" -ge 120 ] || echo "   warning: ${mem_gb} GB of RAM; the recipe is measured on a 128 GB Spark"

case "$PACK" in
  kalibrated|plus)
    MODELS_DIR="${MODELS_DIR:-$HOME/models/deepseek-v4-flash-vision-kalibrated-spark}"; UTIL="${UTIL:-0.925}"
    say "Kalibrated Vision Exp (256 experts, calibrated 3-bit on 28 layers) from $KALIBRATED_REPO -> $MODELS_DIR"
    if [ -f "$MODELS_DIR/tp1/rank-sliced-tp1-manifest.json" ] && [ -f "$MODELS_DIR/dspark-draft-k64/model.safetensors.index.json" ]; then
      say "pack present, skipping the download"
    else
      hf download "$KALIBRATED_REPO" --local-dir "$MODELS_DIR" || die "download failed. If the repo is not reachable yet, PACK=mixedk ./start.sh serves vcruz305's MixedK (docs/KALIBRATED.md says how the Kalibrated pack is rebuilt from the source)"
      [ -f "$MODELS_DIR/tp1/rank-sliced-tp1-manifest.json" ] || die "the download has no tp1/rank-sliced-tp1-manifest.json"
    fi
    ;;
  mixedk)
    MODELS_DIR="${MODELS_DIR:-$HOME/models/deepseek-v4-flash-vision-spark}"; UTIL="${UTIL:-0.88}"
    PACK_DIR="${PACK_DIR:-$HOME/models/dsv4-vision-ablit-exl3-mixedk}"
    say "MixedK (vcruz305: 256 experts, 2-bit + 6 layers at 3-bit) -> $MODELS_DIR"
    if [ -f "$MODELS_DIR/tp1/rank-sliced-tp1-manifest.json" ] && [ -f "$MODELS_DIR/dspark-draft-k64/model.safetensors.index.json" ]; then
      say "converted pack present, skipping download and conversion"
    else
      n=$(ls "$PACK_DIR"/model-000*-of-00048.safetensors 2>/dev/null | wc -l)
      if [ "$n" -eq 48 ]; then say "source pack present ($PACK_DIR)"; else
        say "downloading the vcruz305 pack (~95 GB) from https://huggingface.co/vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK"
        PACK_DIR="$PACK_DIR" "$REPO/scripts/download.sh" || die "download failed"
      fi
      python3 -c "import torch, safetensors" 2>/dev/null || die "the conversion needs python3 with torch and safetensors: pip install torch safetensors"
      say "converting to the served layout (~30 min, idempotent)"
      PACK_DIR="$PACK_DIR" MODELS_DIR="$MODELS_DIR" "$REPO/scripts/convert.sh" || die "conversion failed"
    fi
    ;;
  *) die "PACK must be kalibrated or mixedk (got '$PACK')" ;;
esac

# ---- serve ----------------------------------------------------------------------------------------
"$REPO/scripts/stop.sh" >/dev/null 2>&1 || true
if [ "$PROXY" = 1 ]; then EPORT="$ENGINE_PORT"; else EPORT="$PORT"; fi
say "starting the engine (util $UTIL, context $CTX, engine port $EPORT)"
MODELS_DIR="$MODELS_DIR" UTIL="$UTIL" CTX="$CTX" PORT="$EPORT" CONTAINER="$CONTAINER" "$REPO/scripts/serve.sh"

if [ "$RAMWATCH" = 1 ]; then
  nohup "$REPO/scripts/ramwatch.sh" "$CONTAINER" "$RAMWATCH_FLOOR_MB" >>"$RUN_DIR/ramwatch.log" 2>&1 &
  echo $! >"$RUN_DIR/ramwatch.pid"; say "ramwatch on: stops the container under ${RAMWATCH_FLOOR_MB} MB of MemAvailable ($RUN_DIR/ramwatch.log)"
fi
if [ "$PROXY" = 1 ]; then
  nohup python3 "$REPO/scripts/badwords_proxy.py" --host "${PROXY_HOST:-0.0.0.0}" --listen "$PORT" --upstream "http://127.0.0.1:$ENGINE_PORT" >>"$RUN_DIR/badwords_proxy.log" 2>&1 &
  echo $! >"$RUN_DIR/badwords_proxy.pid"; say "bad_words proxy on :$PORT -> engine :$ENGINE_PORT (every completion request gets the )Skip mask)"
fi
[ "$WAIT" = 1 ] || { say "not waiting; follow with: docker logs -f $CONTAINER"; exit 0; }

say "waiting for http://127.0.0.1:$EPORT/v1/models (cold load ~2 min; the FIRST boot also compiles ~10 min of kernels)"
t0=$(date +%s); last=""
while :; do
  if curl -fsS --max-time 3 "http://127.0.0.1:$EPORT/v1/models" >/dev/null 2>&1; then break; fi
  docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" || { docker logs "$CONTAINER" 2>&1 | tail -n 30; die "the container exited during load (log above; $RUN_DIR/ramwatch.log says if ramwatch stopped it)"; }
  line=$(docker logs "$CONTAINER" 2>&1 | grep -E 'Model loading took|KV cache size|Capturing|compil|Uvicorn|ERROR' | tail -n 1 | cut -c1-140)
  [ "$line" != "$last" ] && [ -n "$line" ] && { echo "   $line"; last="$line"; }
  sleep 10
done
say "ready in $(( $(date +%s) - t0 )) s"
docker logs "$CONTAINER" 2>&1 | grep -E 'Model loading took|GPU KV cache size' | tail -n 2 | sed 's/^/   /'
if [ "$PROXY" = 1 ]; then
  curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && say "proxy answers on :$PORT" || die "the proxy on :$PORT does not answer ($RUN_DIR/badwords_proxy.log)"
fi
cat <<EOT

Try it:
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
    "model": "deepseek-v4-flash-vision-exp",
    "messages": [{"role":"user","content":"Say hello in one line."}], "max_tokens": 40}'
  scripts/vision_probe.py http://127.0.0.1:$PORT deepseek-v4-flash-vision-exp     # an image request
Stop:  scripts/stop.sh        (engine, proxy and ramwatch)
Note:  the proxy adds "bad_words": [")Skip", ",Skip", ".Skip"] to every request (docs/DECODE_PATH_TOKEN_LEAK.md);
       with PROXY=0 add it yourself.
EOT
