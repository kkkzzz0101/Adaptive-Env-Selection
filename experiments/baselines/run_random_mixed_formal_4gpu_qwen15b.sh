#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/adaptive env selection'

export MODEL_PATH=${MODEL_PATH:-/root/models/Qwen2.5-1.5B}
export MODEL_TAG=${MODEL_TAG:-qwen15b}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-160}
export ROLLOUT_N=${ROLLOUT_N:-4}
export BASELINE_STEPS=${BASELINE_STEPS:-1600}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export SAVE_FREQ=${SAVE_FREQ:-200}
export TEST_FREQ=${TEST_FREQ:-200}

bash "$ROOT/experiments/baselines/run_random_mixed_formal_4gpu.sh"
