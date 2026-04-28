# DURCL (Automated Distribution-Level Rebucketing Curriculum Learning)

Current codebase status: **Math+Zebra course-project experiments + DUMP trainer + DURCL scheduler module**.

## What This Repo Runs Now
- **Training core**: `references/DUMP/verl` (used for random baseline training runs).
- **Data/protocol for the course report**: Math+Zebra mixed reasoning tasks.
- **Scoring**:
  - SEC path patches in `references/sec/verl/verl/...`
  - DUMP path custom scorer routing in `references/DUMP/verl/utils/reward_score/`
- **Scheduler code**: `src/scheduler/` (kept in repo; baseline run can be done without scheduler).

## Key Files
- Random baseline launch (DUMP): `scripts/run_baseline_random_dump_2gpu.sh`
- DUMP scorer router: `references/DUMP/verl/utils/reward_score/__init__.py`
- DUMP sec4 scorer: `references/DUMP/verl/utils/reward_score/sec4_mixed.py`
- DUMP trainer patch: `references/DUMP/verl/trainer/ppo/ray_trainer.py`
- Progress notes: `docs/BASELINE_PROGRESS.md`

## Quick Start (Clone and Run)
```bash
git clone https://github.com/kkkzzz0101/Adaptive-Env-Selection.git
cd Adaptive-Env-Selection

# Example env overrides
export CUDA_VISIBLE_DEVICES=0,1
export N_GPUS_PER_NODE=2
export MODEL_PATH=/root/models/Qwen2.5-1.5B-Instruct
export TOTAL_TRAINING_STEPS=1000
export TEST_FREQ=200

bash scripts/run_baseline_random_dump_2gpu.sh
```

For larger machines, increase `CUDA_VISIBLE_DEVICES` and `N_GPUS_PER_NODE` (script supports env overrides).

## Current Experiment Results (Difficulty-init + Rebucket)

### Step-200 validate comparison
- random baseline:
  - `math_train = 0.460`
  - `zebra_train = 0.250`
- no-rebucket scheduler:
  - `math_train = 0.480`
  - `zebra_train = 0.338`

### Rebucket highlights
- rebucket 100 -> 300 (step 300 validate):
  - `math_train = 0.520`
  - `zebra_train = 0.287`
- rebucket step 200 validate:
  - `math_train = 0.540`
  - `zebra_train = 0.287`

### Baseline 200 -> 300 reference
- final:
  - `math_train = 0.560`
  - `zebra_train = 0.275`

### Accuracy-init no-rebucket validation
- step 100:
  - `math_train = 0.480`
  - `zebra_train = 0.263`
- step 200:
  - `math_train = 0.500`
  - `zebra_train = 0.300`

### Accuracy-init rebucket validation
- step 200:
  - `math_train = 0.540`
  - `zebra_train = 0.300`

## Main Analysis Direction

- Even with coarse difficulty-only initialization, rebucketing already shows meaningful structural correction signals.
- Accuracy-based initialization plus rebucketing improves Math over the no-rebucket run while matching Zebra at step 200.

## Detailed Report
- `docs/result_report.md`

## Ongoing Work
- Multi-seed validation and task-balance diagnostics remain needed before making a broad final-performance claim.

## Notes
- Runtime artifacts (`checkpoints/`, `logs/`, `outputs/`) are intentionally git-ignored.
- This repo tracks only DURCL-related hotfix files under `references/DUMP/` rather than the full vendor tree.
