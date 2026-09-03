#!/usr/bin/env bash
# Download the source pack (~95 GB, 48 shards). The repo is gated: accept the terms on
# https://huggingface.co/vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK first and `hf auth login`.
# hf_xet (pip install hf_xet) makes this several times faster on a Spark; do not disable it.
set -euo pipefail
PACK_DIR="${PACK_DIR:-$HOME/models/dsv4-vision-ablit-exl3-mixedk}"
command -v hf >/dev/null || { echo "install the CLI: pip install -U huggingface_hub hf_xet" >&2; exit 2; }
hf download vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK --local-dir "$PACK_DIR"
n=$(ls "$PACK_DIR"/model-000*-of-00048.safetensors 2>/dev/null | wc -l)
echo "shards: $n/48 in $PACK_DIR"
[ "$n" -eq 48 ]
