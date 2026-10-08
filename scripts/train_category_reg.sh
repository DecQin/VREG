#!/bin/bash
# Convergence experiment: Category-aligned REG
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export MLP_WORKER_GPU=${MLP_WORKER_GPU:-2}
export MLP_WORKER_0_HOST=127.0.0.1
export MLP_WORKER_0_PORT=29501
export MLP_ROLE_INDEX=0
export MLP_WORKER_NUM=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PYTHONPATH:/data1/qjyuan/OmniNWM
export PYTHONPATH=/data1/qjyuan/OmniNWM/omninwm/models/ops/occ_pooling:$PYTHONPATH

SEED=${1:-42}

# Check resume
OUTPUT_DIR="./outputs/convergence/category_reg/seed_${SEED}"
LOAD_ARG=""
if [ -f "${OUTPUT_DIR}/latest" ]; then
    LATEST=$(cat "${OUTPUT_DIR}/latest")
    if [ -d "${OUTPUT_DIR}/checkpoints/${LATEST}" ]; then
        LOAD_ARG="--load ${OUTPUT_DIR}/checkpoints/${LATEST}"
        echo "[RESUME] Found checkpoint: ${LOAD_ARG}"
    fi
fi

torchrun --nproc-per-node $MLP_WORKER_GPU \
  --master-addr $MLP_WORKER_0_HOST \
  --master-port $MLP_WORKER_0_PORT \
  --node-rank $MLP_ROLE_INDEX \
  --nnodes $MLP_WORKER_NUM \
  tools/train_convergence.py \
  configs/train/convergence/category_reg.py \
  --seed $SEED \
  $LOAD_ARG
