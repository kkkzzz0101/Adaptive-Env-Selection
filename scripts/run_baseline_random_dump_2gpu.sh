#!/usr/bin/env bash
set -euo pipefail

cd '/root/adaptive env selection'
source /home/vipuser/miniconda3/bin/activate aes

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH='/root/adaptive env selection/references/DUMP:'"${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

DATA_ROOT='/root/adaptive env selection/experiments/baselines/data_sec4_2gpu_1k/mixed'
TRAIN_FILE="$DATA_ROOT/train.parquet"
VAL_FILE="$DATA_ROOT/val.parquet"
MATH500_FILE="$DATA_ROOT/math500_test.parquet"

MODEL_PATH=${MODEL_PATH:-/root/models/Qwen2.5-1.5B-Instruct}
EXP_NAME=${EXP_NAME:-baseline_random_dump_2gpu_1k}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1152}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
ROLLOUT_N=${ROLLOUT_N:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1000}
SAVE_FREQ=${SAVE_FREQ:-200}
TEST_FREQ=${TEST_FREQ:-200}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
export AES_REWARD_NUM_EXAMINE=${AES_REWARD_NUM_EXAMINE:-0}
export AES_VAL_REWARD_NUM_EXAMINE=${AES_VAL_REWARD_NUM_EXAMINE:-0}

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=64 \
  data.enable_curriculum_learning=False \
  data.data_source_key=null \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=0 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  +actor_rollout_ref.rollout.micro_batch_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  critic.model.path="$MODEL_PATH" \
  critic.model.tokenizer_path="$MODEL_PATH" \
  critic.model.use_remove_padding=False \
  critic.model.enable_gradient_checkpointing=False \
  critic.optim.lr=1e-5 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.forward_micro_batch_size_per_gpu=1 \
  critic.ppo_max_token_len_per_gpu=12288 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger="['console']" \
  trainer.project_name='aes_baseline_dump' \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=1 \
  trainer.default_local_dir="checkpoints/aes_baseline_dump/$EXP_NAME" \
  trainer.default_hdfs_dir=null \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.total_epochs=1 \
  +trainer.val_before_train=False \
  trainer.total_training_steps=$TOTAL_TRAINING_STEPS

echo "[INFO] Training finished. MATH500 file ready at: $MATH500_FILE"
