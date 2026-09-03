#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash-Vision-Exp (EXL3 MixedK, converted to rank-sliced tp1) on one DGX Spark.
#
# This is the exact `docker run` measured in the README: the 0xSero sparkinfer image, the
# MiaAI-Lab 256k entrypoint, and this repo's overlay (K2 guards + the vision port) mounted
# read-only over the image's Python files. No image rebuild.
#
#   MODELS_DIR   converted model root (tp1/, dspark-draft-k64/)   default ~/models/deepseek-v4-flash-vision-spark
#   CTX          max_model_len                                     default 245760
#   UTIL         gpu_memory_utilization                           default 0.88  (0.86 also verified; >0.92 starves the driver)
#   MODE         dspark | mtp0 | mtp2 | mtp3                        default dspark (K5 draft); mtp0 = no speculation
#   PORT         engine port                                       default 30021
#   CONTAINER    docker name                                       default dsvision-spark
#   SERVED_MODEL_NAME                                              default deepseek-v4-flash-vision-exp
#   EXTRA_VLLM_ARGS                                                default: see below (mm cache off, encoder attn SDPA)
#   DROP_CACHES=1  run `sync; echo 1 > drop_caches` via sudo before boot (page cache counts against CUDA-free on UMA)
#   DRY_RUN=1      print the command and exit
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4"
BASE="$REPO/third_party/MiaAI-Lab-DeepSeek-v4-Flash-One-DGX-Spark/image-patch"
OVERLAY="$REPO/overlay/dsvision"

MODELS_DIR="${MODELS_DIR:-$HOME/models/deepseek-v4-flash-vision-spark}"
CTX="${CTX:-245760}"
UTIL="${UTIL:-0.88}"
MODE="${MODE:-dspark}"
PORT="${PORT:-30021}"
CONTAINER="${CONTAINER:-dsvision-spark}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---long-prefill-token-threshold 4096 --mm-processor-cache-gb 0 --mm-encoder-attn-backend TORCH_SDPA}"
DSPARK_TOKENS="${DSPARK_TOKENS:-5}"

case "$MODE" in dspark|mtp0|mtp2|mtp3) ;; *) echo "MODE must be dspark|mtp0|mtp2|mtp3 (got '$MODE')" >&2; exit 2 ;; esac

# ---- preflight -------------------------------------------------------------------------------
TP1="$MODELS_DIR/tp1"
[ -f "$TP1/rank-sliced-tp1-manifest.json" ] || { echo "no converted model at $TP1 — run scripts/convert.sh first" >&2; exit 2; }
arch=$(python3 -c "import json;print(json.load(open('$TP1/config.json'))['architectures'][0])")
if [ "$arch" != "DeepseekV4ForConditionalGeneration" ]; then
  echo "config.json architecture is $arch: the checkpoint is in the text-only view." >&2
  echo "run: TP1_DIR=$TP1 $REPO/tools/use_vision.sh on" >&2; exit 2
fi
if [ "$MODE" = dspark ] && [ ! -f "$MODELS_DIR/dspark-draft-k64/model.safetensors.index.json" ]; then
  echo "MODE=dspark but no draft at $MODELS_DIR/dspark-draft-k64 — scripts/convert.sh builds it" >&2; exit 2
fi
mkdir -p "$MODELS_DIR/cache" "$MODELS_DIR/hf-empty"

avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
if [ "${DROP_CACHES:-0}" = 1 ]; then sync; echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null; avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); fi
if [ "$avail_mb" -lt 112000 ]; then
  echo "WARNING: MemAvailable is ${avail_mb} MB; the load needs ~112 GB free on unified memory." >&2
  echo "         stop other engines, then DROP_CACHES=1 (page cache of the last model counts against CUDA)." >&2
fi

# ---- mounts ----------------------------------------------------------------------------------
MOUNTS=(
  -v "$MODELS_DIR:/models"
  -v "$MODELS_DIR/cache:/cache"
  -v "$MODELS_DIR/hf-empty:/hf-cache"
  # MiaAI-Lab recipe layer (MIT): 256k entrypoint, xgrammar boot fix, launcher, DSpark draft class,
  # SM120 MLA prefill dispatcher, tiny-decode kernel, tp1 coalescer (unused: we ship tp1 directly)
  -v "$BASE/entrypoint-toolfix.sh:/patch-run-entrypoint.sh:ro"
  -v "$BASE/entrypoint-256k.sh:/opt/recipe/scripts/entrypoint.sh:ro"
  -v "$BASE/serve-ds4-flash.sh:/opt/vllm/serve-ds4-flash.sh:ro"
  -v "$BASE/coalesce_rank_sliced_exl3.py:/opt/recipe/scripts/coalesce_rank_sliced_exl3.py:ro"
  -v "$BASE/vllm/models/deepseek_v4/nvidia/dspark.py:/opt/vllm/vllm/models/deepseek_v4/nvidia/dspark.py:ro"
  -v "$BASE/sparkinfer/moe/_shared/kernels/tiny_decode.py:/opt/sparkinfer/sparkinfer/moe/_shared/kernels/tiny_decode.py:ro"
  -v "$BASE/sparkinfer/attention/_shared/mla/prefill.py:/opt/sparkinfer/sparkinfer/attention/_shared/mla/prefill.py:ro"
  -v "$BASE/sparkinfer/attention/_shared/mla/prefill_mg.py:/opt/sparkinfer/sparkinfer/attention/_shared/mla/prefill_mg.py:ro"
)
# this repo's overlay: every .py under overlay/dsvision/{sparkinfer,vllm} lands on the same path in the image
while IFS= read -r f; do MOUNTS+=( -v "$OVERLAY/$f:/opt/sparkinfer/$f:ro" ); done < <(cd "$OVERLAY" && find sparkinfer -name '*.py' | sort)
while IFS= read -r f; do MOUNTS+=( -v "$OVERLAY/$f:/opt/vllm/$f:ro" );       done < <(cd "$OVERLAY" && find vllm -name '*.py' | sort)

# ---- environment (the MiaAI-Lab 256k launcher reads all of these) --------------------------
ENV=(
  -e HF_TOKEN="${HF_TOKEN:-}" -e HF_HOME=/hf-cache
  -e PORT="$PORT" -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME"
  # the entrypoint would download+coalesce the 0xSero pack if /models/tp1 had no manifest; ours has one, so these are inert
  -e MODEL_REPO=0xSero/deepseek-v4-flash-0731-spark
  -e MODEL_REVISION=22f28d32b9b29b4352eaa380ff8c2c170b2847ab
  -e MODEL_SOURCE_DIR=/hf-cache/hub/models--0xSero--deepseek-v4-flash-0731-spark/snapshots/22f28d32b9b29b4352eaa380ff8c2c170b2847ab
  -e MAX_MODEL_LEN="$CTX" -e MAX_NUM_SEQS=1 -e MAX_NUM_BATCHED_TOKENS=4096
  -e VLLM_DSV4_PADDED_NVFP4=0 -e KV_FP8_ROPE=0
  -e MODE="$MODE" -e DSPARK_TOKENS="$DSPARK_TOKENS" -e DSPARK_CAPACITY=0
  -e DSPARK_DYNAMIC_DRAFT_DEPTH=0 -e DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW=8
  -e REJECTION_SAMPLE_METHOD=standard -e DRAFT_SAMPLE_METHOD=probabilistic
  -e DSPARK_DRAFT_EXPERTS=64 -e DSPARK_STRUCTURED_EXPERTS_PER_CATEGORY=32
  -e VLLM_USE_B12X_WO_PROJECTION=1
  -e GPU_MEMORY_UTILIZATION="$UTIL"
  -e VERIFY_MODEL_CHECKSUMS="${VERIFY_MODEL_CHECKSUMS:-0}"
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=6 -e CUDAGRAPH_CAPTURE_SIZES=6 -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
  -e EXTRA_VLLM_ARGS="$EXTRA_VLLM_ARGS"
  -e DEFAULT_CHAT_TEMPLATE_KWARGS_THINKING="${THINKING:-true}" -e DEFAULT_CHAT_TEMPLATE_KWARGS_EFFORT="${EFFORT:-max}"
  -e VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=4096
  -e KV_OFFLOAD_GB=0 -e VLLM_USE_SIMPLE_KV_OFFLOAD=0
)

CMD=(docker run -d --name "$CONTAINER" --restart on-failure:1 --gpus all --ipc host --network host --shm-size 16g
  --entrypoint /bin/bash
  --health-cmd "curl -fsS --max-time 5 http://127.0.0.1:$PORT/health" --health-interval 30s --health-timeout 5s
  --health-start-period 20m --health-retries 3
  "${ENV[@]}" "${MOUNTS[@]}" "$IMAGE" -lc 'bash /patch-run-entrypoint.sh && exec /opt/recipe/scripts/entrypoint.sh')

if [ "${DRY_RUN:-0}" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
"${CMD[@]}"
echo "started $CONTAINER — follow with: docker logs -f $CONTAINER"
echo "ready when: curl -s http://127.0.0.1:$PORT/v1/models   (cold load ~2 min from NVMe, ~25 s with a warm page cache; first boot also compiles ~10 min of kernels into $MODELS_DIR/cache)"
