# New Environment Quickstart (Qwen2.5-1.5B Baseline)

## 1) Clone repo

```bash
git clone https://github.com/kkkzzz0101/Adaptive-Env-Selection.git
cd Adaptive-Env-Selection
```

## 2) Place references and model

Expected paths:

- `/root/adaptive env selection/references/DUMP`
- `/root/adaptive env selection/references/sec`
- `/root/models/Qwen2.5-1.5B`

## 3) Setup env

```bash
bash scripts/setup_env.sh aes
# if pandas/pyarrow are missing:
conda run -n aes pip install pandas pyarrow
```

## 4) Scheduler smoke

```bash
bash experiments/baselines/run_scheduler_smoke.sh
```

## 5) Run Qwen1.5B 4-GPU baseline

```bash
bash experiments/baselines/run_random_mixed_formal_4gpu_qwen15b.sh
```

Optional override example:

```bash
MODEL_PATH=/root/models/Qwen2.5-1.5B \
BASELINE_STEPS=2000 \
SAVE_FREQ=250 \
TEST_FREQ=250 \
bash experiments/baselines/run_random_mixed_formal_4gpu_qwen15b.sh
```

## 6) Checkpoint location

Saved under:

- `OUT_DIR_BASE/$RUN_NAME`

The run scripts print final `CHECKPOINT_DIR` after launch.
