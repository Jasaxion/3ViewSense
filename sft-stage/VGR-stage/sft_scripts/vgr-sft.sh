#!/usr/bin/env bash
set -euo pipefail

# export WANDB_DISABLED=true
export SWANLAB_LOG_DIR="./swanlab"
export SWANLAB_SAVE_DIR="./swanlab"
export SWANLAB_MODE="local"
PROJECT_NAME="3ViewSense-VGR"

MODEL_QWEN3_VL_4B='Qwen3-VL-4B-Instruct-OMS-SFT'

BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
LR=1e-05
NUM_EPOCHS=1.0
SAVE_STEPS=100000
DATASET_DIR="./OrthoMind-3D/train"
DATASET_NAME="3viewsense-vgr-sft-data"
DEEPSEED_Z3_JSON="./ds_z3_config.json"

lf_train_sft() {
  local model_name="$1"
  local run_name="$2"
  local template="$3"
  local model_path="$4"

  llamafactory-cli train \
    --stage sft \
    --deepspeed "$DEEPSEED_Z3_JSON" \
    --do_train True \
    --model_name_or_path "$model_path" \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template "$template" \
    --flash_attn auto \
    --dataset_dir "$DATASET_DIR" \
    --dataset "$DATASET_NAME" \
    --cutoff_len 16384 \
    --group_by_length True \
    --packing False \
    --max_new_tokens 16384 \
    --learning_rate "$LR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --max_samples 30000 \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --gradient_checkpointing True \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps "$SAVE_STEPS" \
    --warmup_ratio 0.1 \
    --enable_thinking False \
    --output_dir "./output-stage2/${model_name}/${run_name}" \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --freeze_vision_tower True \
    --freeze_multi_modal_projector True \
    --freeze_language_model False \
    --image_max_pixels 262144 \
    --image_min_pixels 1024 \
    --video_max_pixels 16384 \
    --video_min_pixels 256 \
    --use_swanlab True \
    --swanlab_project "$PROJECT_NAME" \
    --swanlab_run_name "$run_name" \
    --swanlab_mode local
}

lf_train_sft "Qwen3-VL-4B-Instruct"   "VGR-Qwen3_VL_4B"    "qwen3_vl_nothink" "$MODEL_QWEN3_VL_4B"
