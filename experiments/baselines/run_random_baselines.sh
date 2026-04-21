#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/adaptive env selection'
DUMP_ROOT="$ROOT/references/DUMP"
CONDA='/home/vipuser/miniconda3/bin/conda'
MODEL_PATH='/root/models/Qwen2.5-0.5B'

DATA_ROOT="$ROOT/experiments/baselines/data"
LOG_ROOT="$ROOT/experiments/baselines/logs"
OUT_DIR_BASE=${OUT_DIR_BASE:-/root/dump_baseline_random}
SEEDS=${SEEDS:-42}
BASELINE_STEPS=${BASELINE_STEPS:-30}
BASELINE_EPOCHS=${BASELINE_EPOCHS:-1}
MATH_TRAIN_SAMPLES=${MATH_TRAIN_SAMPLES:-32}
MATH_VAL_SAMPLES=${MATH_VAL_SAMPLES:-8}

mkdir -p "$DATA_ROOT" "$LOG_ROOT"

$CONDA run -n aes python "$ROOT/scripts/prepare_random_baseline_datasets.py" \
  --kk-train-source "$ROOT/experiments/dump_smoke/data/train.parquet" \
  --kk-val-source "$ROOT/experiments/dump_smoke/data/val.parquet" \
  --out-root "$DATA_ROOT" \
  --math-train-samples "$MATH_TRAIN_SAMPLES" \
  --math-val-samples "$MATH_VAL_SAMPLES" \
  --seed 42

for seed in $SEEDS; do
  for setup in kk_only math_only mixed; do
    TRAIN_FILE="$DATA_ROOT/$setup/train.parquet"
    VAL_FILE="$DATA_ROOT/$setup/val.parquet"
    RUN_NAME="qwen05b_random_${setup}_seed${seed}"
    LOG_FILE="$LOG_ROOT/${RUN_NAME}.log"

    echo "[RUN] $RUN_NAME"
    (
      cd "$DUMP_ROOT"
      export CUDA_VISIBLE_DEVICES=0
      export VLLM_ATTENTION_BACKEND=XFORMERS
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

      $CONDA run -n aes python -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="$TRAIN_FILE" \
        data.val_files="$VAL_FILE" \
        data.train_batch_size=1 \
        data.enable_curriculum_learning=False \
        data.data_source_key=null \
        data.max_prompt_length=1024 \
        data.max_response_length=64 \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.ppo_mini_batch_size=1 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=2048 \
        actor_rollout_ref.rollout.name=hf \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=0 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=2048 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=999 \
        trainer.logger="['console']" \
        trainer.project_name='baseline_random' \
        trainer.experiment_name="$RUN_NAME" \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.default_local_dir="$OUT_DIR_BASE/$RUN_NAME" \
        trainer.default_hdfs_dir=null \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs="$BASELINE_EPOCHS" \
        trainer.total_training_steps="$BASELINE_STEPS"
    ) 2>&1 | tee "$LOG_FILE"

    echo "[DONE] $RUN_NAME -> $LOG_FILE"
  done
done


echo '[ALL DONE] random baseline runs completed.'
