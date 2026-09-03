#!/usr/bin/env bash
# Pack (HF-sharded, BF16 non-routed, 48 shards) -> 0xSero rank-sliced tp1 layout + DSpark K64 draft + vision view.
# Idempotent: finished steps are skipped. ~25 min for the conversion on a USB-3 source disk, minutes for the rest.
#
#   PACK_DIR    the downloaded pack                  default ~/models/dsv4-vision-ablit-exl3-mixedk
#   MODELS_DIR  output root                          default ~/models/deepseek-v4-flash-vision-spark
#   PYTHON      interpreter with torch + safetensors default python3
#   VERIFY      sample size for verify_tp1 (0 skips) default 300
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACK_DIR="${PACK_DIR:-$HOME/models/dsv4-vision-ablit-exl3-mixedk}"
MODELS_DIR="${MODELS_DIR:-$HOME/models/deepseek-v4-flash-vision-spark}"
PY="${PYTHON:-python3}"
OUT="$MODELS_DIR/tp1"; DRAFT="$MODELS_DIR/dspark-draft-k64"
T="$REPO/tools"; REF="$REPO/reference"
BUILD_DRAFT="$REPO/third_party/MiaAI-Lab-DeepSeek-v4-Flash-One-DGX-Spark/image-patch/build_dspark_draft.py"

n=$(ls "$PACK_DIR"/model-000*-of-00048.safetensors 2>/dev/null | wc -l)
[ "$n" -eq 48 ] || { echo "pack incomplete: $n/48 shards in $PACK_DIR" >&2; exit 2; }
[ -f "$PACK_DIR/config.json" ] || { echo "missing $PACK_DIR/config.json" >&2; exit 2; }
"$PY" -c "import torch, safetensors" 2>/dev/null || { echo "$PY needs torch and safetensors (pip install torch safetensors)" >&2; exit 2; }
mkdir -p "$OUT"

echo "== 1/6 convert experts + requantize non-routed tensors -> $OUT"
if [ -f "$OUT/rank-sliced-tp1-manifest.json" ]; then echo "   manifest exists, skip"; else
  "$PY" "$T/convert_vision_pack.py" --src "$PACK_DIR" --ref-spec "$REF/ref_spec.json" --out "$OUT"
fi

echo "== 2/6 config.json (text view; rms_norm_eps forced to 1e-6 for the fused mHC kernel)"
"$PY" "$T/make_config.py" "$PACK_DIR/config.json" "$REF/0xsero-tp1-config.json" "$OUT" "$OUT/bitrates.json"
cp "$PACK_DIR/tokenizer.json" "$PACK_DIR/tokenizer_config.json" "$OUT/" 2>/dev/null || true
[ -f "$PACK_DIR/generation_config.json" ] && cp "$PACK_DIR/generation_config.json" "$OUT/" || true

echo "== 3/6 DSpark K64 draft (built from the text view, before the vision tensors are added)"
if [ -f "$DRAFT/model.safetensors.index.json" ]; then echo "   draft exists, skip"; else
  rm -rf "$DRAFT"
  "$PY" "$BUILD_DRAFT" --source "$OUT" --output "$DRAFT" --experts 64 --structured-per-category 32 --plan "$REF/draft_plan.json"
fi

echo "== 4/6 vision view: keep text copies, append the 316 vision-side tensors, write config-vision.json"
for f in config.json model.safetensors.index.json rank-sliced-tp1-manifest.json; do
  b="${f%.json}.text.json"; [ -f "$OUT/$b" ] || cp "$OUT/$f" "$OUT/$b"
done
if [ -f "$OUT/model.safetensors.index.vision.json" ]; then echo "   vision index exists, skip"; else
  cp "$OUT/model.safetensors.index.text.json" "$OUT/model.safetensors.index.vision.json"
  cp "$OUT/rank-sliced-tp1-manifest.text.json" "$OUT/rank-sliced-tp1-manifest.vision.json"
  IDX=model.safetensors.index.vision.json MAN=rank-sliced-tp1-manifest.vision.json "$PY" "$T/add_vision_tensors.py" "$PACK_DIR" "$OUT"
fi
"$PY" "$T/make_vision_config.py" "$PACK_DIR/config.json" "$OUT"

echo "== 5/6 switch the checkpoint to the vision view"
TP1_DIR="$OUT" bash "$T/use_vision.sh" on

if [ "${VERIFY:-300}" != 0 ]; then
  echo "== 6/6 verify against the source pack (names, dtypes, shapes; byte-equality on ${VERIFY:-300} sampled experts)"
  "$PY" "$T/verify_tp1.py" "$PACK_DIR" "$OUT" "${VERIFY:-300}"
fi
echo "== done"; du -sh "$OUT" "$DRAFT"
"$PY" - "$OUT" <<'PY'
import json, sys
o = sys.argv[1]; idx = json.load(open(f"{o}/model.safetensors.index.json"))
c = json.load(open(f"{o}/config.json"))
print("tensors:", len(idx["weight_map"]), " total GiB:", round(idx["metadata"]["total_size"] / 2**30, 2),
      " arch:", c["architectures"][0], " experts:", c["n_routed_experts"], " k_values:", c["hybrid_tr3_tail"]["k_values"])
PY
