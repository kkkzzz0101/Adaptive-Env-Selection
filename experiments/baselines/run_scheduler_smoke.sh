#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/adaptive env selection'
CONDA='/home/vipuser/miniconda3/bin/conda'
OUT_JSON=${OUT_JSON:-$ROOT/experiments/baselines/results_1024/scheduler_smoke_report.json}

$CONDA run -n aes python "$ROOT/scripts/smoke_scheduler.py" \
  --train-parquet "$ROOT/experiments/baselines/data_formal/mixed/train.parquet" \
  --num-clusters ${NUM_CLUSTERS:-3} \
  --steps ${STEPS:-180} \
  --batch-size ${BATCH_SIZE:-32} \
  --max-per-source ${MAX_PER_SOURCE:-20} \
  --seed ${SEED:-42} \
  --out-json "$OUT_JSON"

echo "[DONE] scheduler smoke report -> $OUT_JSON"
