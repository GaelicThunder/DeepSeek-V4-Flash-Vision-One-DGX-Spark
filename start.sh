#!/usr/bin/env bash
# start.sh — one command from a fresh DGX Spark to a serving endpoint.
#
#   ./start.sh                 # MixedK+ if its pack is published (PLUS_REPO), otherwise vcruz305's MixedK:
#                              #   download -> (convert) -> serve -> wait for /v1/models -> print a test request
#   PACK=mixedk ./start.sh     # force the MixedK route: download the vcruz305 pack (~95 GB), convert (~30 min), serve
#   PACK=plus   ./start.sh     # force MixedK+: download the ready-to-serve pack (~100 GB), serve
#   ./start.sh --no-wait       # return right after `docker run`
#
# Tunables (environment):
#   PLUS_REPO   Hub repo of the ready-to-serve MixedK+ pack (tp1/ + dspark-draft-k64/).  Empty = not published yet.
#   MODELS_DIR  where the served pack lives      default ~/models/deepseek-v4-flash-vision-spark   (-plus for MixedK+)
#   PACK_DIR    where the vcruz305 pack is downloaded (MixedK route)   default ~/models/dsv4-vision-ablit-exl3-mixedk
#   CTX         max_model_len                    default 245760
#   UTIL        gpu_memory_utilization           default 0.88 (MixedK) / 0.925 (MixedK+); never above 0.93 on 128 GB
#   PORT        engine port                      default 30021
#   HF_TOKEN    Hub token (both packs are gated: accept the terms on the model pages first, or `hf auth login`)
# Everything else (mounts, overlay, engine flags) is scripts/serve.sh, unchanged.
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUS_REPO="${PLUS_REPO:-}"
PACK="${PACK:-auto}"
WAIT=1; [ "${1:-}" = "--no-wait" ] && WAIT=0
PORT="${PORT:-30021}"; CTX="${CTX:-245760}"

say(){ printf '\033[36m==\033[0m %s\n' "$*"; }
die(){ printf '\033[31mxx\033[0m %s\n' "$*" >&2; exit 2; }

# ---- preflight ----------------------------------------------------------------------------------
command -v docker >/dev/null || die "docker is required"
docker info >/dev/null 2>&1 || die "docker is installed but not usable by this user (docker group? daemon running?)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/   gpu: /' || echo "   (nvidia-smi not found; continuing)"
command -v hf >/dev/null || die "the Hub CLI is required: pip install -U 'huggingface_hub[hf_xet]'   (hf_xet makes the download several times faster)"
mem_gb=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo); [ "$mem_gb" -ge 120 ] || echo "   warning: ${mem_gb} GB of RAM; the recipe is measured on a 128 GB Spark"

if [ "$PACK" = auto ]; then PACK=$([ -n "$PLUS_REPO" ] && echo plus || echo mixedk); fi
case "$PACK" in
  plus)
    [ -n "$PLUS_REPO" ] || die "MixedK+ is not published as a download yet (PLUS_REPO is empty). Until it is: PACK=mixedk ./start.sh serves vcruz305's MixedK, and scripts/aplus/ rebuilds MixedK+ from the source (2×H200, ~4 h). See docs/APLUS.md."
    MODELS_DIR="${MODELS_DIR:-$HOME/models/deepseek-v4-flash-vision-spark-plus}"; UTIL="${UTIL:-0.925}"
    say "MixedK+ (256 experts, 3-bit on 28 layers) from $PLUS_REPO -> $MODELS_DIR"
    if [ -f "$MODELS_DIR/tp1/rank-sliced-tp1-manifest.json" ] && [ -f "$MODELS_DIR/dspark-draft-k64/model.safetensors.index.json" ]; then
      say "pack present, skipping the download"
    else
      hf download "$PLUS_REPO" --local-dir "$MODELS_DIR" || die "download failed (gated repo? accept the terms on https://huggingface.co/$PLUS_REPO and set HF_TOKEN or run: hf auth login)"
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
        say "downloading the vcruz305 pack (~95 GB, gated: accept the terms on https://huggingface.co/vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK)"
        PACK_DIR="$PACK_DIR" "$REPO/scripts/download.sh" || die "download failed"
      fi
      python3 -c "import torch, safetensors" 2>/dev/null || die "the conversion needs python3 with torch and safetensors: pip install torch safetensors"
      say "converting to the served layout (~30 min, idempotent)"
      PACK_DIR="$PACK_DIR" MODELS_DIR="$MODELS_DIR" "$REPO/scripts/convert.sh" || die "conversion failed"
    fi
    ;;
  *) die "PACK must be auto, plus or mixedk (got '$PACK')" ;;
esac

# ---- serve ----------------------------------------------------------------------------------------
say "starting the engine (util $UTIL, context $CTX, port $PORT)"
MODELS_DIR="$MODELS_DIR" UTIL="$UTIL" CTX="$CTX" PORT="$PORT" "$REPO/scripts/serve.sh"
CONTAINER="${CONTAINER:-dsvision-spark}"
[ "$WAIT" = 1 ] || { say "not waiting; follow with: docker logs -f $CONTAINER"; exit 0; }

say "waiting for http://127.0.0.1:$PORT/v1/models (cold load ~2 min; the FIRST boot also compiles ~10 min of kernels)"
t0=$(date +%s); last=""
while :; do
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then break; fi
  docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" || { docker logs "$CONTAINER" 2>&1 | tail -n 30; die "the container exited during load (log above)"; }
  line=$(docker logs "$CONTAINER" 2>&1 | grep -E 'Model loading took|KV cache size|Capturing|compil|Uvicorn|ERROR' | tail -n 1 | cut -c1-140)
  [ "$line" != "$last" ] && [ -n "$line" ] && { echo "   $line"; last="$line"; }
  sleep 10
done
say "ready in $(( $(date +%s) - t0 )) s"
docker logs "$CONTAINER" 2>&1 | grep -E 'Model loading took|GPU KV cache size' | tail -n 2 | sed 's/^/   /'
cat <<EOF

Try it:
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
    "model": "deepseek-v4-flash-vision-exp",
    "messages": [{"role":"user","content":"Say hello in one line."}],
    "max_tokens": 40, "bad_words": [")Skip", ",Skip", ".Skip"]}'
  scripts/vision_probe.py http://127.0.0.1:$PORT deepseek-v4-flash-vision-exp     # an image request
Stop:  scripts/stop.sh
Note:  keep "bad_words": [")Skip", ",Skip", ".Skip"] in requests, or put scripts/badwords_proxy.py in front (docs/DECODE_PATH_TOKEN_LEAK.md).
EOF
