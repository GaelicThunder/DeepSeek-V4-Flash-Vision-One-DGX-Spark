#!/usr/bin/env bash
# pod setup: download the abliterated source and build exllamav3 (0531096 + the conversion patches in this directory) in parallel
set -u -o pipefail
cd /workspace; mkdir -p logs
export DEBIAN_FRONTEND=noninteractive
echo "=== $(date -Is) apt/pip"
apt-get update -qq > logs/apt.log 2>&1; apt-get install -y -qq rsync ninja-build git >> logs/apt.log 2>&1
pip install -q -U "huggingface_hub[hf_xet]" hf_transfer pyyaml safetensors ninja > logs/pip.log 2>&1
echo "=== $(date -Is) download start"
( hf download drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit --local-dir /workspace/src-ablit > logs/download.log 2>&1; echo "### DOWNLOAD exit $? $(date -Is)" >> logs/download.log ) &
echo "=== $(date -Is) exllamav3 build start"
git clone -q https://github.com/turboderp-org/exllamav3 /workspace/exl3 > logs/build.log 2>&1 && cd /workspace/exl3 && git checkout -q 0531096 && git apply /workspace/tools/exl3-conversion.patch && echo "patch applied" >> /workspace/logs/build.log \
 && pip install -q -r requirements.txt >> /workspace/logs/build.log 2>&1 \
 && TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=32 pip install -v --no-build-isolation . >> /workspace/logs/build.log 2>&1
echo "### BUILD exit $? $(date -Is)" >> /workspace/logs/build.log
cd /workspace/exl3 && python -c 'import exllamav3, torch; from exllamav3 import ext; print("exllamav3 import OK, torch", torch.__version__, "cuda", torch.cuda.device_count())' >> /workspace/logs/build.log 2>&1
echo "=== $(date -Is) setup done (download may still run)"
