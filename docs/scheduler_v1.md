# Scheduler V1 (Difficulty + UCB + Re-bucketing)

This repository now includes a standalone scheduler module:

- `src/scheduler/adaptive_curriculum_scheduler.py`
- `scripts/smoke_scheduler.py`

## Method scope (v1)

- Calibrated initial difficulty clusters (`easy/medium/hard` by default)
- Cluster-level UCB sampling with softmax + probability floor
- Per-sample decayed state tracking with `|advantage|`
- Warm-up + low-frequency neighbor re-bucketing
- No semantic propagation and no dynamic full reclustering

## Why checkpoint was previously missing

Previous training scripts set:

- `trainer.save_freq=-1`

That explicitly disables checkpoint saving.

## Current checkpoint policy

The baseline training scripts now expose checkpoint knobs:

- `SAVE_FREQ` (default enabled)
- `TEST_FREQ` (default enabled)

Main scripts:

- `experiments/baselines/run_random_mixed_formal_4gpu.sh`
- `experiments/baselines/run_random_mixed_formal.sh`
- `experiments/baselines/run_random_mixed_formal_1024.sh`
- `experiments/baselines/run_random_baselines.sh`

## Quick smoke

```bash
bash experiments/baselines/run_scheduler_smoke.sh
```

## 4-GPU 1600-step baseline

```bash
bash experiments/baselines/run_random_mixed_formal_4gpu.sh
```

Outputs and checkpoints are stored under:

- `OUT_DIR_BASE/$RUN_NAME`
