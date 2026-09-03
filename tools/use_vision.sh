#!/usr/bin/env bash
# Switch the tp1 checkpoint between the text-only view and the vision view.
set -euo pipefail
T="${TP1_DIR:-$HOME/models/deepseek-v4-flash-vision-spark/tp1}"
case "${1:-status}" in
  on)
    [ -f "$T/config.text.json" ] || cp "$T/config.json" "$T/config.text.json"
    [ -f "$T/model.safetensors.index.text.json" ] || cp "$T/model.safetensors.index.json" "$T/model.safetensors.index.text.json"
    [ -f "$T/rank-sliced-tp1-manifest.text.json" ] || cp "$T/rank-sliced-tp1-manifest.json" "$T/rank-sliced-tp1-manifest.text.json"
    cp "$T/config-vision.json" "$T/config.json"
    cp "$T/model.safetensors.index.vision.json" "$T/model.safetensors.index.json"
    cp "$T/rank-sliced-tp1-manifest.vision.json" "$T/rank-sliced-tp1-manifest.json"
    echo "vision view ON" ;;
  off)
    cp "$T/config.text.json" "$T/config.json"
    cp "$T/model.safetensors.index.text.json" "$T/model.safetensors.index.json"
    cp "$T/rank-sliced-tp1-manifest.text.json" "$T/rank-sliced-tp1-manifest.json"
    echo "vision view OFF (text-only)" ;;
  *)
    n=$(python3 -c "import json;print(len(json.load(open('$T/model.safetensors.index.json'))['weight_map']))")
    a=$(python3 -c "import json;print(json.load(open('$T/config.json'))['architectures'][0])")
    echo "arch=$a tensors=$n" ;;
esac
