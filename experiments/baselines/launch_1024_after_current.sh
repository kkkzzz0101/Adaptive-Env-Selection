#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/adaptive env selection'
LOG_WAIT="$ROOT/experiments/baselines/run_random_mixed_formal_1024_wait.log"
LOG_RUN="$ROOT/experiments/baselines/run_random_mixed_formal_1024.nohup.log"

mkdir -p "$ROOT/experiments/baselines"

echo "[$(date '+%F %T')] waiting current 512/64 run to finish..." >> "$LOG_WAIT"
while pgrep -f "trainer.experiment_name=qwen05b_random_mixed_formal_seed42" >/dev/null 2>&1; do
  sleep 20
done

echo "[$(date '+%F %T')] current run finished, start 1024/96 run" >> "$LOG_WAIT"
cd "$ROOT"
bash experiments/baselines/run_random_mixed_formal_1024.sh >> "$LOG_RUN" 2>&1
