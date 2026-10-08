#!/bin/bash
# Evaluate all convergence checkpoints
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$ROOT_DIR"

SEEDS=${1:-42}
VARIANTS="baseline,category_reg,frame_reg"
STEPS="500,1000,2000,5000,10000,20000,25000,30000"
GT_DIR=${2:-""}

EXTRA_ARGS=""
if [ -n "$GT_DIR" ]; then
    EXTRA_ARGS="--gt_dir $GT_DIR"
fi

echo "============================================"
echo " Convergence Evaluation"
echo " Seeds: $SEEDS"
echo " Variants: $VARIANTS"
echo " Steps: $STEPS"
echo "============================================"

python tools/evaluate_convergence.py \
    --root outputs/convergence \
    --variants $VARIANTS \
    --seeds $SEEDS \
    --steps $STEPS \
    $EXTRA_ARGS

echo "[DONE] Evaluation complete."
