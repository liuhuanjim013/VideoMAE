#!/bin/bash

# Distributed Finetuning Example for VideoMAE on Kinetics-400 5k-filtered (156 classes)
#
# Usage:
#   On node 0 (master):
#     bash finetune_distributed.sh 0 127.0.0.1
#   On node 1 (worker):
#     bash finetune_distributed.sh 1 127.0.0.1
#
# NOTE:
#   - You must have an SSH tunnel open from node 1 to node 0 to forward the master port (12320).
#     Example: ssh -N -L 12320:localhost:12320 user@<master-node-ip>
#   - Adjust DATA_PATH, MODEL_PATH, OUTPUT_DIR, and LOG_DIR as needed.

# Set paths
DATA_PATH="/home/ubuntu/videomae/k400-top5k-filtered"
MODEL_PATH="/home/ubuntu/.cache/huggingface/hub/models--MCG-NJU--videomae-base/snapshots/dc740ceda42fce44faed2ea03c6d447db72f6af9/model.safetensors"
DATETIME=$(date +%Y%m%d%H%M)
OUTPUT_DIR="./output/kinetics_finetune-5k-filtered-${DATETIME}"
LOG_DIR="./logs/kinetics_finetune-5k-filtered-${DATETIME}"

# Distributed parameters
NNODES=2
NODE_RANK=$1
MASTER_ADDR=$2
MASTER_PORT=12320

# Training parameters
MODEL=vit_base_patch16_224
DATASET=Kinetics-400
NB_CLASSES=156
BATCH_SIZE=16
UPDATE_FREQ=1
INPUT_SIZE=224
SHORT_SIDE_SIZE=224
NUM_FRAMES=16
SAMPLING_RATE=4
OPT=adamw
WEIGHT_DECAY=0.05
LR=1e-3
OPT_BETAS="0.9 0.999"
EPOCHS=320
WARMUP_LR=1e-06
SAVE_CKPT_FREQ=10
NUM_WORKERS=16

# WANDB parameters (optional)
USE_WANDB=1
WANDB_PROJECT="videomae-kinetics400"
WANDB_RUN_NAME="videomae_base_kinetics_5k-filtered-${DATETIME}_distributed"
WANDB_ENTITY="your-wandb-entity"

# Run distributed training
OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=1 \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_port=${MASTER_PORT} \
    --master_addr=${MASTER_ADDR} \
    run_class_finetuning.py \
    --model ${MODEL} \
    --data_set ${DATASET} \
    --nb_classes ${NB_CLASSES} \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --batch_size ${BATCH_SIZE} \
    --update_freq ${UPDATE_FREQ} \
    --input_size ${INPUT_SIZE} \
    --short_side_size ${SHORT_SIDE_SIZE} \
    --num_frames ${NUM_FRAMES} \
    --sampling_rate ${SAMPLING_RATE} \
    --opt ${OPT} \
    --weight_decay ${WEIGHT_DECAY} \
    --lr ${LR} \
    --opt_betas ${OPT_BETAS} \
    --epochs ${EPOCHS} \
    --warmup_lr ${WARMUP_LR} \
    --save_ckpt_freq ${SAVE_CKPT_FREQ} \
    --num_workers ${NUM_WORKERS} \
    --dist_eval \
    $( [[ $USE_WANDB -eq 1 ]] && echo "--use_wandb --wandb_project \"${WANDB_PROJECT}\" --wandb_run_name \"${WANDB_RUN_NAME}\" --wandb_entity \"${WANDB_ENTITY}\"" ) 