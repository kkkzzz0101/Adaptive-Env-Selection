#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-aes}"
PROJECT_ROOT="/root/adaptive env selection"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found"
  exit 1
fi

# Create env by cloning base so CUDA-enabled torch is inherited.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[INFO] conda env '$ENV_NAME' already exists, skip clone"
else
  echo "[INFO] creating conda env '$ENV_NAME' by cloning base"
  conda create -n "$ENV_NAME" --clone base -y
fi

# Repair pip if clone produced inconsistent pip internals.
SITE_PKGS="/home/vipuser/miniconda3/envs/${ENV_NAME}/lib/python3.12/site-packages"
if [ -d "$SITE_PKGS" ]; then
  rm -rf "$SITE_PKGS/pip" "$SITE_PKGS"/pip-*.dist-info || true
fi
conda run -n "$ENV_NAME" python -m ensurepip --upgrade

# Install minimal inference dependencies.
conda run -n "$ENV_NAME" python -m pip install -U -r "$PROJECT_ROOT/requirements.infer.txt"

echo "[INFO] done. test with: conda run -n $ENV_NAME python '$PROJECT_ROOT/scripts/smoke_test_qwen.py'"
