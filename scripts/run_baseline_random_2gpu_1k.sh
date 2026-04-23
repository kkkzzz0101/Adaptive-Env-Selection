#!/usr/bin/env bash
set -euo pipefail

cd '/root/adaptive env selection'
source /home/vipuser/miniconda3/bin/activate aes

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH='/root/adaptive env selection/references/sec/verl:'"${PYTHONPATH:-}"

DATA_ROOT='/root/adaptive env selection/experiments/baselines/data_sec4_2gpu_1k/mixed'
TRAIN_FILE="$DATA_ROOT/train.parquet"
VAL_FILE="$DATA_ROOT/val.parquet"
MATH500_FILE="$DATA_ROOT/math500_test.parquet"

MODEL_PATH=${MODEL_PATH:-/root/models/Qwen2.5-1.5B-Instruct}
EXP_NAME=${EXP_NAME:-baseline_random_2gpu_1k}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1152}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="['$TRAIN_FILE']" \
  data.val_files="['$VAL_FILE']" \
  data.train_batch_size=32 \
  data.val_batch_size=64 \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size=4 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.n_val=4 \
  actor_rollout_ref.rollout.top_k=0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.model.path="$MODEL_PATH" \
  critic.model.use_remove_padding=False \
  critic.model.enable_gradient_checkpointing=False \
  critic.optim.lr=1e-5 \
  critic.ppo_micro_batch_size=4 \
  critic.forward_micro_batch_size=4 \
  critic.ppo_max_token_len_per_gpu=16384 \
  trainer.sec.enable=True \
  trainer.sec.strategy=random \
  trainer.total_training_steps=1000 \
  trainer.save_freq=200 \
  trainer.test_freq=200 \
  trainer.critic_warmup=0 \
  trainer.logger="['console']" \
  trainer.project_name='aes_baseline' \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="checkpoints/aes_baseline/$EXP_NAME" \
  trainer.val_before_train=False

echo "[INFO] Training finished. MATH500 file ready at: $MATH500_FILE"
