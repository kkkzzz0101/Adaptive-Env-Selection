#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/adaptive env selection'
SEC_ROOT="$ROOT/references/sec"
CONDA='/home/vipuser/miniconda3/bin/conda'
MODEL_PATH=${MODEL_PATH:-/root/models/Qwen2.5-1.5B}
MODEL_TAG=${MODEL_TAG:-qwen15b}

DATA_ROOT="$ROOT/experiments/baselines/data_formal"
LOG_ROOT="$ROOT/experiments/baselines/logs"
OUT_DIR_BASE=${OUT_DIR_BASE:-/root/sec_baseline_random_mixed}
SEED=${SEED:-42}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-768}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-160}
ROLLOUT_N=${ROLLOUT_N:-4}
BASELINE_STEPS=${BASELINE_STEPS:-1600}
BASELINE_EPOCHS=${BASELINE_EPOCHS:-1}

SAVE_FREQ=${SAVE_FREQ:-200}
TEST_FREQ=${TEST_FREQ:-200}

RUN_NAME=${RUN_NAME:-${MODEL_TAG}_random_sec4_formal_p${MAX_PROMPT_LENGTH}_r${MAX_RESPONSE_LENGTH}_n${ROLLOUT_N}_s${BASELINE_STEPS}_seed${SEED}}

mkdir -p "$DATA_ROOT" "$LOG_ROOT"

$CONDA run -n aes python "$ROOT/scripts/prepare_sec4_random_dataset.py" \
  --sec-root "$ROOT/references/sec/data" \
  --out-root "$DATA_ROOT" \
  --countdown-levels '1,2,3,4' \
  --zebra-levels '1,2,3,4' \
  --arc-levels '1,2,3,4' \
  --math-levels '1,2,3,4,5' \
  --train-per-bucket 240 \
  --val-per-bucket 30 \
  --test-per-bucket 80 \
  --math-train-per-level 300 \
  --math-val-per-level 30 \
  --seed "$SEED"

TRAIN_FILE="$DATA_ROOT/mixed/train.parquet"
VAL_FILE="$DATA_ROOT/mixed/val.parquet"
LOG_FILE="$LOG_ROOT/${RUN_NAME}.log"

(
  cd "$SEC_ROOT"
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
  export VLLM_ATTENTION_BACKEND=XFORMERS
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  $CONDA run -n aes python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=1 \
    data.enable_curriculum_learning=False \
    data.data_source_key=null \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1536 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1536 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1536 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=999 \
    trainer.logger="['console']" \
    trainer.project_name='baseline_random_sec4' \
    trainer.experiment_name="$RUN_NAME" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.default_local_dir="$OUT_DIR_BASE/$RUN_NAME" \
    trainer.default_hdfs_dir=null \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$BASELINE_EPOCHS" \
    trainer.total_training_steps="$BASELINE_STEPS"
) 2>&1 | tee "$LOG_FILE"

echo "[DONE] $RUN_NAME -> $LOG_FILE"
echo "[CHECKPOINT_DIR] $OUT_DIR_BASE/$RUN_NAME"
