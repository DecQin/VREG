#!/usr/bin/env python3
"""Extract GT RGB/depth/seg images for a given inference sample index.

Usage:
    python tools/extract_gt_images.py --index 51 --output_dir outputs/infer_nusc1/gt_images
    python tools/extract_gt_images.py --index 51,52,53 --output_dir outputs/infer_nusc1/gt_images
"""
import os
import sys
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmengine.config import Config
from omninwm.datasets.datasets import NuscenesVideoDataset, PANDASET_CAM_TO_NUSCENES


def save_depth_vis(depth_np, save_path, max_depth=100.0):
    """Save depth as colorized visualization."""
    depth_norm = np.clip(depth_np / max_depth, 0, 1)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
    # Also save raw npz
    raw_path = str(save_path).replace('.png', '.npz')
    np.savez_compressed(raw_path, depth=depth_np)
    cv2.imwrite(str(save_path), depth_color)


def save_seg_vis(seg_np, save_path, seg_color_map=None):
    """Save segmentation as colorized visualization + raw."""
    # Save raw
    raw_path = str(save_path).replace('.png', '_raw.png')
    cv2.imwrite(raw_path, seg_np)
    # Save colorized if color map provided
    if seg_color_map is not None:
        seg_color = np.zeros((*seg_np.shape[:2], 3), dtype=np.uint8)
        for seg_id in np.unique(seg_np):
            color = seg_color_map.get(seg_id, [0, 0, 0])
            seg_color[seg_np[:, :, 0] == seg_id] = color
        cv2.imwrite(str(save_path), seg_color)
    else:
        cv2.imwrite(str(save_path), seg_np)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True, help="Sample index(es), comma-separated")
    parser.add_argument("--output_dir", type=str, default="outputs/gt_images")
    parser.add_argument("--config", type=str, default="configs/inference/ablation_infer.py")
    parser.add_argument("--start_index", type=int, default=0, help="start_index used in inference (default: 0)")
    parser.add_argument("--max_depth", type=float, default=100.0)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    ds_cfg = dict(cfg.dataset)
    ds_cfg.pop("type", None)

    print(f"Building dataset from {ds_cfg.get('pkl_path', '?')} ...")
    dataset = NuscenesVideoDataset(**ds_cfg)
    print(f"Dataset size: {len(dataset)}")

    indices = [int(i) for i in args.index.split(",")]

    for idx in indices:
        actual_idx = idx - args.start_index
        if actual_idx < 0 or actual_idx >= len(dataset.data):
            print(f"Index {idx} (actual {actual_idx}) out of range, skipping")
            continue

        sample = dataset.data[actual_idx]
        paths = sample.get("path", [])
        tokens = sample.get("token", [])
        dataset_name = getattr(dataset, 'dataset_name', 'nuscenes')
        depth_seg_root = getattr(dataset, 'depth_seg_root', None)

        sample_dir = os.path.join(args.output_dir, f"{args.start_index}_{idx:04d}")
        rgb_dir = os.path.join(sample_dir, "rgb")
        depth_dir = os.path.join(sample_dir, "depth")
        seg_dir = os.path.join(sample_dir, "seg")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)

        for f_idx, frame_paths in enumerate(paths):
            if isinstance(frame_paths, str):
                frame_paths = [frame_paths]
            frame_tokens = tokens[f_idx] if tokens and f_idx < len(tokens) else []
            view_order = list(dataset.view_order) if hasattr(dataset, 'view_order') else []

            for v_idx, img_path in enumerate(frame_paths):
                # Get view name
                cam_name = os.path.basename(os.path.dirname(img_path))
                view_name = view_order[v_idx] if v_idx < len(view_order) else cam_name
                token = frame_tokens[v_idx] if v_idx < len(frame_tokens) else ""

                # --- Save RGB ---
                resolved_path = img_path
                if not os.path.exists(resolved_path):
                    resolved_path = dataset._resolve_image_path(img_path)
                if os.path.exists(resolved_path):
                    ext = os.path.splitext(resolved_path)[1]
                    dst = os.path.join(rgb_dir, f"frame{f_idx:03d}_{view_name}{ext}")
                    shutil.copy2(resolved_path, dst)
                else:
                    print(f"  WARNING: RGB not found: {resolved_path}")

                # --- Save Depth ---
                try:
                    if dataset_name == "pandaset" and depth_seg_root:
                        parts = img_path.split('/')
                        ps_idx = parts.index('pandaset')
                        scene_id = parts[ps_idx + 1]
                        cn = parts[ps_idx + 3]
                        frame_name = Path(img_path).stem
                        nusc_view = PANDASET_CAM_TO_NUSCENES.get(cn, cn)
                        pt = f"{scene_id}_{int(frame_name):06d}"
                        depth_path = Path(depth_seg_root) / scene_id / nusc_view / pt / "refined_depth.npz"
                    else:
                        # nuScenes
                        depth_path = Path('data/nuscenes_12hz_depth_unzip') / token / view_name / "refined_depth.npz"

                    if depth_path.exists():
                        depth_np = np.load(str(depth_path))['depth_pred']
                        save_depth_vis(depth_np, os.path.join(depth_dir, f"frame{f_idx:03d}_{view_name}.png"),
                                       max_depth=args.max_depth)
                    else:
                        print(f"  WARNING: Depth not found: {depth_path}")
                except Exception as e:
                    print(f"  WARNING: Depth error frame{f_idx} {view_name}: {e}")

                # --- Save Segmentation ---
                try:
                    if dataset_name == "pandaset" and depth_seg_root:
                        seg_path = Path(depth_seg_root) / scene_id / nusc_view / pt / "seg.png"
                    else:
                        # nuScenes
                        seg_path_str = img_path.replace(
                            '/nuscenes/', '/nuscenes_seg/'
                        ).replace('.jpg', '.png').replace('/samples/', '/samples_seg/').replace('/sweeps/', '/sweeps_seg/')
                        seg_path = Path(seg_path_str)

                    if seg_path.exists():
                        seg_np = cv2.imread(str(seg_path))
                        if seg_np is not None:
                            save_seg_vis(seg_np, os.path.join(seg_dir, f"frame{f_idx:03d}_{view_name}.png"),
                                         seg_color_map=getattr(dataset, 'seg_color_map', None))
                        else:
                            print(f"  WARNING: Seg read failed: {seg_path}")
                    else:
                        print(f"  WARNING: Seg not found: {seg_path}")
                except Exception as e:
                    print(f"  WARNING: Seg error frame{f_idx} {view_name}: {e}")

        print(f"Saved GT (rgb/depth/seg) for sample {idx} -> {sample_dir}")


if __name__ == "__main__":
    main()
