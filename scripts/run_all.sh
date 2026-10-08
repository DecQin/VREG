#!/bin/bash
# Run all convergence training sequentially (avoids GPU contention)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$ROOT_DIR"

SEEDS=${1:-42}
GPU_DEVICES=${2:-0,1}

echo "============================================"
echo " Convergence Comparison Experiment"
echo " Seeds: $SEEDS"
echo " GPUs: $GPU_DEVICES"
echo "============================================"

export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

for SEED in $(echo $SEEDS | tr ',' ' '); do
    echo ""
    echo "============================================"
    echo " Seed: $SEED"
    echo "============================================"

    # 1. Baseline
    echo "[1/3] Training baseline (seed=$SEED)..."
    bash scripts/convergence/train_baseline.sh $SEED

    # 2. Category-aligned REG
    echo "[2/3] Training category_reg (seed=$SEED)..."
    bash scripts/convergence/train_category_reg.sh $SEED

    # 3. Frame-aligned REG
    echo "[3/3] Training frame_reg (seed=$SEED)..."
    bash scripts/convergence/train_frame_reg.sh $SEED
done

echo ""
echo "============================================"
echo " All training complete!"
echo "============================================"
