#!/bin/bash
# Plot convergence results
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$ROOT_DIR"

TARGET_FID=${1:-11.3}

echo "============================================"
echo " Convergence Plotting (target FID: $TARGET_FID)"
echo "============================================"

python scripts/plot_convergence.py \
    --results outputs/convergence/eval_results/convergence_results.json \
    --output_dir outputs/convergence/figures \
    --target_fid $TARGET_FID

echo "[DONE] Plots saved to outputs/convergence/figures/"
