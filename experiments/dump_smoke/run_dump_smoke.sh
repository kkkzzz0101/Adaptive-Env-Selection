#!/usr/bin/env bash
set -euo pipefail

cd '/root/adaptive env selection/references/DUMP'
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_FILE='/root/adaptive env selection/experiments/dump_smoke/data/train.parquet'
VAL_FILE='/root/adaptive env selection/experiments/dump_smoke/data/val.parquet'
MODEL_PATH='/root/models/Qwen2.5-0.5B'
OUT_DIR='/root/dump_smoke_output'
MATH_TRAIN_SAMPLES=${MATH_TRAIN_SAMPLES:-24}
MATH_VAL_SAMPLES=${MATH_VAL_SAMPLES:-8}

# Build a mixed smoke dataset: existing KK subset + MATH-500 subset.
/home/vipuser/miniconda3/bin/conda run -n aes python '/root/adaptive env selection/scripts/build_dump_math500_smoke.py' \
  --kk-train "$TRAIN_FILE" \
  --kk-val "$VAL_FILE" \
  --out-train "$TRAIN_FILE" \
  --out-val "$VAL_FILE" \
  --math-train-samples "$MATH_TRAIN_SAMPLES" \
  --math-val-samples "$MATH_VAL_SAMPLES" \
  --seed 42

/home/vipuser/miniconda3/bin/conda run -n aes python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=1 \
  data.enable_curriculum_learning=True \
  data.data_source_key=data_source \
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
  trainer.project_name='dump_smoke' \
  trainer.experiment_name='qwen05b_dump_scheduler_loop' \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.default_local_dir="$OUT_DIR" \
  trainer.default_hdfs_dir=null \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1
