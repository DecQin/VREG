"""
Unified convergence evaluation script.
Loads checkpoints for each variant × seed × step, generates videos
with fixed seeds and validation clips, computes FID/FVD, and outputs
convergence_results.csv and convergence_results.json.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def find_checkpoints(root, variant, seed, steps):
    """Find checkpoint directories for given variant/seed at specified steps."""
    ckpt_dir = os.path.join(root, variant, f"seed_{seed}", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print(f"[WARN] Checkpoint dir not found: {ckpt_dir}")
        return {}

    found = {}
    for item in sorted(os.listdir(ckpt_dir)):
        item_path = os.path.join(ckpt_dir, item)
        if not os.path.isdir(item_path):
            continue
        # Parse step from directory name (e.g., epoch0-global_step500)
        try:
            global_step = int(item.split("global_step")[-1])
        except (ValueError, IndexError):
            continue
        if global_step in steps:
            found[global_step] = item_path
    return found


def load_metadata(root, variant, seed):
    """Load run_metadata.json for a variant/seed."""
    path = os.path.join(root, variant, f"seed_{seed}", "run_metadata.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_timing(root, variant, seed, ckpt_path):
    """Load timing data from checkpoint."""
    timing_path = os.path.join(ckpt_path, "convergence_timing.json")
    if os.path.isfile(timing_path):
        with open(timing_path) as f:
            return json.load(f)
    return {}


def run_inference_for_checkpoint(cfg_path, ckpt_path, save_dir, seed,
                                  num_samples=50, device_id=0):
    """Run inference for a single checkpoint using the existing inference script."""
    os.makedirs(save_dir, exist_ok=True)
    cmd = [
        sys.executable, "-u", "tools/inference_reg.py",
        cfg_path,
        "--load", ckpt_path,
        "--save_dir", save_dir,
        "--seed", str(seed),
        "--num_samples", str(num_samples),
        "--device", f"cuda:{device_id}",
    ]
    print(f"[EVAL] Running inference: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(device_id)})
    if result.returncode != 0:
        print(f"[ERROR] Inference failed for {ckpt_path}")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return False
    return True


def run_evaluation(pred_dir, gt_dir, output_json):
    """Run FID/FVD evaluation using evaluate.py."""
    cmd = [
        sys.executable, "-u", "tools/evaluate.py",
        "--pred_dir", pred_dir,
        "--gt_dir", gt_dir,
        "--output", output_json,
    ]
    print(f"[EVAL] Running evaluation: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Evaluation failed for {pred_dir}")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return None
    if os.path.isfile(output_json):
        with open(output_json) as f:
            return json.load(f)
    return None


def compute_fid_fvd_simple(pred_dir, gt_dir):
    """
    Compute FID/FVD using evaluate.py logic.
    Falls back to direct computation if subprocess fails.
    """
    # Try using evaluate.py directly
    output_json = os.path.join(pred_dir, "eval_results.json")
    result = run_evaluation(pred_dir, gt_dir, output_json)
    if result is not None:
        return result.get("fid", None), result.get("fvd", None)
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Convergence evaluation")
    parser.add_argument("--root", default="outputs/convergence",
                        help="Root directory for convergence outputs")
    parser.add_argument("--variants", default="baseline,category_reg,frame_reg",
                        help="Comma-separated variant names")
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated seeds")
    parser.add_argument("--steps", default="500,1000,2000,5000,10000,20000,25000,30000",
                        help="Comma-separated checkpoint steps to evaluate")
    parser.add_argument("--gt_dir", default=None,
                        help="Ground truth directory for FID/FVD")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of validation clips to generate per checkpoint")
    parser.add_argument("--cfg_base", default="configs/train/convergence",
                        help="Base config directory")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference, only compute FID/FVD from existing outputs")
    parser.add_argument("--target_fid", type=float, default=11.3,
                        help="Target FID for threshold computation")
    parser.add_argument("--device_id", type=int, default=0,
                        help="GPU device for inference")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    steps = [int(s.strip()) for s in args.steps.split(",")]

    results = []

    for variant in variants:
        cfg_path = os.path.join(args.cfg_base, f"{variant}.py")
        if not os.path.isfile(cfg_path):
            print(f"[WARN] Config not found: {cfg_path}, skipping {variant}")
            continue

        for seed in seeds:
            metadata = load_metadata(args.root, variant, seed)
            ckpt_map = find_checkpoints(args.root, variant, seed, steps)
            print(f"[INFO] {variant}/seed_{seed}: found {len(ckpt_map)}/{len(steps)} checkpoints")

            for step in sorted(steps):
                record = {
                    "variant": variant,
                    "seed": seed,
                    "step": step,
                }

                if step not in ckpt_map:
                    print(f"[WARN] No checkpoint at step {step} for {variant}/seed_{seed}")
                    record.update({"fid": None, "fvd": None, "status": "missing_checkpoint"})
                    results.append(record)
                    continue

                ckpt_path = ckpt_map[step]
                record["checkpoint_path"] = ckpt_path

                # Load timing data
                timing = load_timing(args.root, variant, seed, ckpt_path)
                record["elapsed_hours"] = round(timing.get("elapsed_seconds", 0) / 3600, 4)
                record["gpu_hours"] = timing.get("gpu_hours", None)

                # Load metadata
                record["trainable_params"] = metadata.get("trainable_parameters", None)
                record["reg_params"] = metadata.get("reg_extra_parameters", None)
                record["config_hash"] = metadata.get("config_hash", None)

                # Inference output dir
                pred_dir = os.path.join(
                    args.root, variant, f"seed_{seed}", "eval_outputs", f"step_{step}"
                )

                # Run inference
                if not args.skip_inference:
                    success = run_inference_for_checkpoint(
                        cfg_path, ckpt_path, pred_dir, seed,
                        num_samples=args.num_samples,
                        device_id=args.device_id,
                    )
                    if not success:
                        record.update({"fid": None, "fvd": None, "status": "inference_failed"})
                        results.append(record)
                        continue

                # Compute FID/FVD
                gt_dir = args.gt_dir
                if gt_dir and os.path.isdir(pred_dir):
                    fid, fvd = compute_fid_fvd_simple(pred_dir, gt_dir)
                    record["fid"] = fid
                    record["fvd"] = fvd
                    record["status"] = "ok"
                else:
                    record["fid"] = None
                    record["fvd"] = None
                    record["status"] = "no_gt_dir" if not gt_dir else "no_pred_dir"

                # Count generated clips
                if os.path.isdir(pred_dir):
                    num_clips = len(glob.glob(os.path.join(pred_dir, "**/*.mp4"), recursive=True))
                    record["num_clips"] = num_clips
                else:
                    record["num_clips"] = 0

                results.append(record)
                print(f"[EVAL] {variant}/seed_{seed}/step_{step}: "
                      f"FID={record.get('fid')}, FVD={record.get('fvd')}")

    # == Save results ==
    output_dir = os.path.join(args.root, "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(output_dir, "convergence_results.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"[SAVED] {csv_path}")

    # JSON
    json_path = os.path.join(output_dir, "convergence_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] {json_path}")

    # == Threshold computation ==
    compute_threshold(results, args.target_fid, output_dir)


def compute_threshold(results, target_fid, output_dir):
    """Compute threshold summary: first step where FID <= target_fid."""
    from collections import defaultdict

    # Group by variant
    variant_data = defaultdict(list)
    for r in results:
        if r.get("fid") is not None:
            variant_data[r["variant"]].append(r)

    threshold_rows = []
    baseline_first_step = None

    for variant in sorted(variant_data.keys()):
        records = sorted(variant_data[variant], key=lambda x: x["step"])
        first_observed_step = None
        first_observed_hours = None
        first_observed_gpu_hours = None
        interpolated_step = None

        for i, r in enumerate(records):
            if r["fid"] <= target_fid:
                first_observed_step = r["step"]
                first_observed_hours = r.get("elapsed_hours", None)
                first_observed_gpu_hours = r.get("gpu_hours", None)
                # Linear interpolation for more precise threshold
                if i > 0 and records[i - 1]["fid"] is not None:
                    prev = records[i - 1]
                    fid_prev = prev["fid"]
                    fid_curr = r["fid"]
                    if fid_prev > target_fid and fid_curr <= target_fid:
                        frac = (fid_prev - target_fid) / (fid_prev - fid_curr)
                        step_prev = prev["step"]
                        step_curr = r["step"]
                        interpolated_step = step_prev + frac * (step_curr - step_prev)
                break

        row = {
            "variant": variant,
            "target_fid": target_fid,
            "first_observed_step": first_observed_step,
            "interpolated_step": round(interpolated_step, 1) if interpolated_step else None,
            "wall_clock_hours": first_observed_hours,
            "gpu_hours": first_observed_gpu_hours,
            "step_reduction_vs_baseline": None,
            "wall_clock_speedup_vs_baseline": None,
            "gpu_hour_reduction_vs_baseline": None,
        }

        if variant == "baseline":
            baseline_first_step = first_observed_step
            baseline_hours = first_observed_hours
            baseline_gpu_hours = first_observed_gpu_hours

        threshold_rows.append(row)

    # Compute relative improvements
    for row in threshold_rows:
        if row["first_observed_step"] is not None and baseline_first_step is not None:
            row["step_reduction_vs_baseline"] = round(
                1 - row["first_observed_step"] / baseline_first_step, 4
            )
        if row["wall_clock_hours"] is not None and baseline_hours is not None:
            row["wall_clock_speedup_vs_baseline"] = round(
                baseline_hours / row["wall_clock_hours"], 4
            )
        if row["gpu_hours"] is not None and baseline_gpu_hours is not None:
            row["gpu_hour_reduction_vs_baseline"] = round(
                1 - row["gpu_hours"] / baseline_gpu_hours, 4
            )

    # Save threshold summary
    threshold_path = os.path.join(output_dir, "threshold_summary.csv")
    if threshold_rows:
        fieldnames = list(threshold_rows[0].keys())
        with open(threshold_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(threshold_rows)
        print(f"[SAVED] {threshold_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("THRESHOLD SUMMARY (FID <= {})".format(target_fid))
    print("=" * 70)
    for row in threshold_rows:
        print(f"  {row['variant']:>15s}: step={row['first_observed_step']}, "
              f"hours={row['wall_clock_hours']}, gpu_h={row['gpu_hours']}, "
              f"step_reduction={row['step_reduction_vs_baseline']}, "
              f"speedup={row['wall_clock_speedup_vs_baseline']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
