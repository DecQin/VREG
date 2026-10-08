#!/usr/bin/env python3
"""
Generate depth maps and segmentation maps from PandaSet LiDAR point clouds.

For each camera view, project LiDAR points onto the image plane to create:
1. Sparse depth maps → saved as .npz with key 'depth_pred'
2. Segmentation maps → saved as .png with per-pixel class IDs

Usage:
    python tools/generate_pandaset_depth_seg.py \
        --data_dir /data1/qjyuan/OmniNWM/data/pandaset \
        --output_dir /data1/qjyuan/OmniNWM/data/pandaset_depth_seg \
        --num_workers 8

Output directory structure (per scene):
    <output_dir>/<scene_id>/<view_name>/<frame_id>/
        refined_depth.npz   # depth_pred array [H, W]
        seg.png             # segmentation map [H, W] with class IDs
"""

import argparse
import gzip
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# PandaSet camera name mapping
# ---------------------------------------------------------------------------
PANDASET_CAMERAS = [
    "front_camera",
    "front_left_camera",
    "front_right_camera",
    "left_camera",
    "right_camera",
    "back_camera",
]

PANDASET_TO_NUSCENES = {
    "front_camera":      "CAM_FRONT",
    "front_left_camera": "CAM_FRONT_LEFT",
    "front_right_camera":"CAM_FRONT_RIGHT",
    "left_camera":       "CAM_BACK_LEFT",
    "right_camera":      "CAM_BACK_RIGHT",
    "back_camera":       "CAM_BACK",
}

# PandaSet semseg class IDs → nuScenes-compatible 9-class mapping
# PandaSet 42 classes mapped to our 9-category scheme
PANDASET_SEG_CLASS_MAP = {
    # 0: object (misc static objects)
    1: 0,   # Smoke
    2: 0,   # Exhaust
    3: 0,   # Spray or rain
    4: 0,   # Reflection
    17: 0,  # Towed Object
    20: 0,  # Other Vehicle - Uncommon
    21: 0,  # Other Vehicle - Pedicab
    27: 0,  # Train
    28: 0,  # Trolley
    29: 0,  # Tram / Subway
    32: 0,  # Animals - Bird
    33: 0,  # Animals - Other
    34: 0,  # Pylons
    36: 0,  # Signs
    37: 0,  # Cones
    38: 0,  # Construction Signs
    39: 0,  # Temporary Construction Barriers
    40: 0,  # Rolling Containers
    41: 0,  # Building
    42: 0,  # Other Static Object
    # 1: road
    7: 1,   # Road
    12: 1,  # Driveway
    # 2: sky (no LiDAR points, filled as sky in post-processing)
    # 3: car
    13: 3,  # Car
    14: 3,  # Pickup Truck
    15: 3,  # Medium-sized Truck
    16: 3,  # Semi-truck
    19: 3,  # Other Vehicle - Construction Vehicle
    22: 3,  # Emergency Vehicle
    23: 3,  # Bus
    # 4: pedestrian
    30: 4,  # Pedestrian
    31: 4,  # Pedestrian with Object
    # 5: bicycle/motorcycle
    18: 5,  # Motorcycle
    24: 5,  # Personal Mobility Device
    25: 5,  # Motorized Scooter
    26: 5,  # Bicycle
    # 6: lane marking
    8: 6,   # Lane Line Marking
    9: 6,   # Stop Line Marking
    10: 6,  # Other Road Marking
    # 7: sidewalk
    11: 7,  # Sidewalk
    # 8: vegetation
    5: 8,   # Vegetation
    # 6: ground
    6: 6,   # Ground → lane (gray area)
    35: 6,  # Road Barriers → lane
}

# Colors for segmentation visualization (matching nuScenes convention)
SEG_COLOR_MAP = [
    (0, 0, 0),        # 0: object
    (0, 207, 191),     # 1: road
    (135, 206, 235),   # 2: sky
    (255, 158, 0),     # 3: car
    (0, 0, 230),       # 4: person
    (255, 61, 99),     # 5: bicycle
    (192, 192, 192),   # 6: lane
    (75, 0, 75),       # 7: sidewalk
    (0, 175, 0),       # 8: vegetation
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)    ],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)    ],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
    return R


def pose_to_transform(pose_json: dict) -> np.ndarray:
    """Convert PandaSet pose JSON to 4x4 homogeneous transform."""
    pos = pose_json["position"]
    heading = pose_json["heading"]
    t = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float64)
    q = np.array([heading["w"], heading["x"], heading["y"], heading["z"]], dtype=np.float64)
    R = quaternion_to_rotation_matrix(q)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def project_points_to_camera(
    points_lidar: np.ndarray,
    lidar_pose: np.ndarray,
    cam_pose: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project LiDAR points from world coordinates onto a camera image.

    Args:
        points_lidar: [N, 3] point cloud in LiDAR frame
        lidar_pose: [4, 4] LiDAR-to-world transform
        cam_pose: [4, 4] camera-to-world transform
        K: [3, 3] camera intrinsic matrix
        img_w: image width
        img_h: image height

    Returns:
        u: [M] pixel x coordinates (within image bounds)
        v: [M] pixel y coordinates (within image bounds)
        depths: [M] depth values in camera frame
    """
    # Transform points from LiDAR frame to world frame
    ones = np.ones((points_lidar.shape[0], 1), dtype=points_lidar.dtype)
    pts_h = np.hstack([points_lidar, ones])  # [N, 4]
    pts_world = (lidar_pose @ pts_h.T).T[:, :3]  # [N, 3]

    # Transform from world to camera frame
    cam2world = cam_pose
    world2cam = np.linalg.inv(cam2world)
    pts_cam = (world2cam @ np.hstack([pts_world, ones]).T).T[:, :3]  # [N, 3]

    # Filter points behind the camera
    valid = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid]

    # Project to pixel coordinates
    pts_2d = (K @ pts_cam.T).T  # [N, 3]
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    u = pts_2d[:, 0].astype(np.int32)
    v = pts_2d[:, 1].astype(np.int32)
    depths = pts_cam[:, 2]

    # Filter points within image bounds
    in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u = u[in_img]
    v = v[in_img]
    depths = depths[in_img]

    return u, v, depths


def render_depth_map(
    u: np.ndarray,
    v: np.ndarray,
    depths: np.ndarray,
    img_h: int,
    img_w: int,
    max_depth: float = 100.0,
) -> np.ndarray:
    """
    Render a sparse depth map from projected points.
    Uses z-buffer to handle occlusions (keep closest depth per pixel).

    Args:
        u, v: pixel coordinates
        depths: depth values
        img_h, img_w: image dimensions
        max_depth: maximum depth value for clamping

    Returns:
        depth_map: [H, W] float32 depth map
    """
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)

    # Sort by depth (far to near) so closer points overwrite farther ones
    order = np.argsort(-depths)
    u_sorted = u[order]
    v_sorted = v[order]
    depths_sorted = depths[order]

    # Use z-buffer rendering
    depth_map[v_sorted, u_sorted] = depths_sorted.astype(np.float32)

    # Clamp to max depth
    depth_map[depth_map > max_depth] = max_depth

    return depth_map


def render_seg_map(
    u: np.ndarray,
    v: np.ndarray,
    depths: np.ndarray,
    class_ids: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Render a segmentation map from projected points.
    Uses z-buffer to handle occlusions (keep closest point's class).

    Args:
        u, v: pixel coordinates
        depths: depth values
        class_ids: class IDs per point
        img_h, img_w: image dimensions

    Returns:
        seg_map: [H, W] uint8 segmentation map
    """
    seg_map = np.zeros((img_h, img_w), dtype=np.uint8)

    # Sort by depth (far to near) so closer points overwrite farther ones
    order = np.argsort(-depths)
    u_sorted = u[order]
    v_sorted = v[order]
    class_sorted = class_ids[order]

    seg_map[v_sorted, u_sorted] = class_sorted.astype(np.uint8)

    return seg_map


def process_scene(
    scene_dir: Path,
    output_dir: Path,
    max_depth: float = 100.0,
    skip_existing: bool = True,
) -> int:
    """
    Process a single PandaSet scene: generate depth and segmentation maps
    for all frames and all camera views.

    Args:
        scene_dir: Path to scene directory
        output_dir: Path to output directory
        max_depth: Maximum depth value
        skip_existing: Skip if output already exists

    Returns:
        Number of frames processed
    """
    scene_id = scene_dir.name

    # Load camera intrinsics and poses
    cam_data = {}
    for cam_name in PANDASET_CAMERAS:
        cam_dir = scene_dir / "camera" / cam_name
        if not cam_dir.exists():
            continue

        intr_path = cam_dir / "intrinsics.json"
        poses_path = cam_dir / "poses.json"

        if not intr_path.exists() or not poses_path.exists():
            continue

        with open(intr_path) as f:
            intr = json.load(f)
        with open(poses_path) as f:
            poses = json.load(f)

        K = np.eye(3, dtype=np.float64)
        K[0, 0] = intr["fx"]
        K[1, 1] = intr["fy"]
        K[0, 2] = intr["cx"]
        K[1, 2] = intr["cy"]

        nusc_name = PANDASET_TO_NUSCENES[cam_name]
        cam_data[nusc_name] = {
            "pandaset_name": cam_name,
            "K": K,
            "poses": poses,
        }

    if not cam_data:
        print(f"  WARNING: No cameras found in scene {scene_id}, skipping.")
        return 0

    # Load LiDAR poses
    lidar_poses_path = scene_dir / "lidar" / "poses.json"
    with open(lidar_poses_path) as f:
        lidar_poses = json.load(f)

    num_frames = len(lidar_poses)
    processed = 0

    # Check for semseg availability
    semseg_dir = scene_dir / "annotations" / "semseg"
    has_semseg = semseg_dir.exists() and len(list(semseg_dir.glob("*.pkl.gz"))) > 0

    for frame_idx in range(num_frames):
        # Load LiDAR point cloud
        lidar_path = scene_dir / "lidar" / f"{frame_idx:02d}.pkl.gz"
        if not lidar_path.exists():
            continue

        with gzip.open(lidar_path, "rb") as f:
            lidar_df = pickle.load(f)

        points = lidar_df[["x", "y", "z"]].values.astype(np.float64)

        # Load semseg labels if available
        semseg_ids = None
        if has_semseg:
            semseg_path = semseg_dir / f"{frame_idx:02d}.pkl.gz"
            if semseg_path.exists():
                with gzip.open(semseg_path, "rb") as f:
                    semseg_df = pickle.load(f)
                raw_classes = semseg_df["class"].values.astype(np.int32)
                # Map to our 9-class scheme
                mapped_classes = np.array(
                    [PANDASET_SEG_CLASS_MAP.get(int(c), 0) for c in raw_classes],
                    dtype=np.uint8,
                )
                semseg_ids = mapped_classes

        # Get LiDAR pose (world transform)
        lidar_pose = pose_to_transform(lidar_poses[frame_idx])

        # Project to each camera
        for nusc_name, cd in cam_data.items():
            cam_pose = pose_to_transform(cd["poses"][frame_idx])
            K = cd["K"]

            # Create output directory - use same token format as PKL
            # PKL token format: {scene_id}_{interp_idx:06d}
            # For non-interpolated frames, interp_idx == frame_idx
            token = f"{scene_id}_{frame_idx:06d}"
            out_dir = output_dir / scene_id / nusc_name / token
            out_dir.mkdir(parents=True, exist_ok=True)

            depth_path = out_dir / "refined_depth.npz"
            seg_path = out_dir / "seg.png"

            # Skip if both exist
            if skip_existing and depth_path.exists() and seg_path.exists():
                continue

            # Project points
            u, v, depths = project_points_to_camera(
                points, lidar_pose, cam_pose, K, 1920, 1080,
            )

            # Render depth map
            depth_map = render_depth_map(u, v, depths, 1080, 1920, max_depth)
            np.savez_compressed(str(depth_path), depth_pred=depth_map)

            # Render segmentation map
            if semseg_ids is not None and len(u) > 0:
                # Filter semseg_ids to match valid projected points
                # Need to redo projection to get the valid mask
                ones = np.ones((points.shape[0], 1), dtype=points.dtype)
                pts_h = np.hstack([points, ones])
                pts_world = (lidar_pose @ pts_h.T).T[:, :3]
                cam2world = cam_pose
                world2cam = np.linalg.inv(cam2world)
                pts_cam = (world2cam @ np.hstack([pts_world, ones]).T).T[:, :3]
                valid = pts_cam[:, 2] > 0.1
                # Apply same filtering as project_points_to_camera
                valid_semseg = semseg_ids[valid]
                pts_cam_valid = pts_cam[valid]
                pts_2d = (K @ pts_cam_valid.T).T
                pts_2d[:, 0] /= pts_2d[:, 2]
                pts_2d[:, 1] /= pts_2d[:, 2]
                u_all = pts_2d[:, 0].astype(np.int32)
                v_all = pts_2d[:, 1].astype(np.int32)
                d_all = pts_cam_valid[:, 2]
                in_img = (u_all >= 0) & (u_all < 1920) & (v_all >= 0) & (v_all < 1080)
                seg_map = render_seg_map(
                    u_all[in_img], v_all[in_img], d_all[in_img],
                    valid_semseg[in_img], 1080, 1920,
                )
            else:
                seg_map = np.zeros((1080, 1920), dtype=np.uint8)

            # Save segmentation as PNG
            cv2.imwrite(str(seg_path), seg_map)

        processed += 1

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Generate depth and segmentation maps from PandaSet LiDAR"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory of PandaSet (containing scene folders)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for generated depth/seg maps",
    )
    parser.add_argument(
        "--max_depth", type=float, default=100.0,
        help="Maximum depth value for clamping (default: 100.0)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--scene_ids", type=str, nargs="+", default=None,
        help="Specific scene IDs to process (default: all)",
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="Do not skip existing outputs",
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Discover scenes
    scene_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and (d / "camera").exists()
    ])

    if args.scene_ids is not None:
        scene_dirs = [d for d in scene_dirs if d.name in args.scene_ids]

    print(f"Found {len(scene_dirs)} scenes")
    print(f"Output directory: {output_path}")

    total_processed = 0
    for i, scene_dir in enumerate(scene_dirs):
        print(f"\n[{i+1}/{len(scene_dirs)}] Processing scene {scene_dir.name} ...")
        n = process_scene(
            scene_dir, output_path,
            max_depth=args.max_depth,
            skip_existing=not args.no_skip,
        )
        total_processed += n
        print(f"  Processed {n} frames")

    print(f"\n{'='*60}")
    print(f"Total frames processed: {total_processed}")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
