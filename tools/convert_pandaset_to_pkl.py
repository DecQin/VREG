#!/usr/bin/env python3
"""
Convert PandaSet to nuScenes-compatible PKL format for OmniNWM training.

Usage:
    python tools/convert_pandaset_to_pkl.py \
        --data_dir /path/to/pandaset \
        --output_dir data/pandaset_pkl \
        --interp_hz 12

PandaSet directory structure (per scene):
    <scene_id>/
    ├── annotations/
    │   ├── cuboids/{00..79}.pkl.gz
    │   └── semseg/{00..79}.pkl.gz  (76 scenes only)
    ├── camera/
    │   ├── front_camera/
    │   │   ├── {00..79}.jpg
    │   │   ├── intrinsics.json
    │   │   ├── poses.json
    │   │   └── timestamps.json
    │   ├── front_left_camera/ ...
    │   ├── front_right_camera/ ...
    │   ├── left_camera/ ...
    │   ├── right_camera/ ...
    │   └── back_camera/ ...
    ├── lidar/
    │   ├── {00..79}.pkl.gz
    │   ├── poses.json
    │   └── timestamps.json
    └── meta/
        ├── gps.json
        └── timestamps.json

Output PKL structure (compatible with NuscenesVideoDataset):
    {
        'infos': [
            {
                'token': str,
                'timestamp': float,
                'location': str,
                'timeofday': str,
                'is_key_frame': bool,
                'caption': dict,
                'lidar_path': str,
                'cams': {
                    'CAM_FRONT': {
                        'data_path': str,
                        'type': str,
                        'sample_data_token': str,
                        'sensor2ego_translation': list[3],
                        'sensor2ego_rotation': list[4],   # quaternion [w,x,y,z]
                        'ego2global_translation': list[3],
                        'ego2global_rotation': list[4],    # quaternion [w,x,y,z]
                        'timestamp': float,
                        'sensor2lidar_rotation': ndarray[3,3],
                        'sensor2lidar_translation': ndarray[3],
                        'camera_intrinsics': ndarray[3,3],
                    },
                    ...  # 6 views total
                },
                'lidar2ego_translation': list[3],
                'lidar2ego_rotation': list[4],
                'ego2global_translation': list[3],
                'ego2global_rotation': list[4],
                # placeholders for nuScenes-specific fields
                'gt_boxes': np.zeros((0,7)),
                'gt_names': [],
                'gt_velocity': [],
                'num_lidar_pts': [],
                'num_radar_pts': [],
                'valid_flag': [],
                'gt_box_ids': [],
                'sweeps': [],
            },
            ...
        ],
        'metadata': {'version': 'pandaset_v1'},
        'scene_tokens': [[token1, token2, ...], ...],  # list of scenes, each scene is list of tokens
    }
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

import numpy as np


# ---------------------------------------------------------------------------
# PandaSet ↔ nuScenes view name mapping
# ---------------------------------------------------------------------------
# PandaSet has 6 cameras: front, front_left, front_right, left, right, back
# nuScenes has 6 cameras: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
#                         CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT
# We map PandaSet views to nuScenes convention so the downstream code works.
PANDASET_TO_NUSCENES = {
    "front_camera":      "CAM_FRONT",
    "front_left_camera": "CAM_FRONT_LEFT",
    "front_right_camera":"CAM_FRONT_RIGHT",
    "left_camera":       "CAM_BACK_LEFT",    # PandaSet left → nuScenes back-left
    "right_camera":      "CAM_BACK_RIGHT",   # PandaSet right → nuScenes back-right
    "back_camera":       "CAM_BACK",
}

# The order the dataset class iterates over views
VIEW_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
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


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def pose_to_transform(pose_json: dict) -> np.ndarray:
    """
    Convert PandaSet pose JSON to 4x4 homogeneous transform.
    PandaSet pose format:
        {
            "position": {"x": ..., "y": ..., "z": ...},
            "heading": {"w": ..., "x": ..., "y": ..., "z": ...}  # quaternion
        }
    Returns 4x4 matrix in world coordinates.
    """
    pos = pose_json["position"]
    heading = pose_json["heading"]
    t = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float64)
    q = np.array([heading["w"], heading["x"], heading["y"], heading["z"]], dtype=np.float64)
    R = quaternion_to_rotation_matrix(q)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# PandaSet data loading (without devkit dependency)
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_pkl_gz(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def get_camera_names(scene_dir: Path) -> List[str]:
    """Return list of camera folder names under scene_dir/camera/."""
    cam_dir = scene_dir / "camera"
    if not cam_dir.exists():
        return []
    return sorted([d.name for d in cam_dir.iterdir() if d.is_dir()])


def load_camera_poses(scene_dir: Path, cam_name: str) -> List[dict]:
    """Load poses.json for a given camera."""
    poses_path = scene_dir / "camera" / cam_name / "poses.json"
    return load_json(poses_path)


def load_camera_intrinsics(scene_dir: Path, cam_name: str) -> dict:
    """Load intrinsics.json for a given camera.
    Returns dict with keys like 'fx', 'fy', 'cx', 'cy' or 'focal_length', etc.
    """
    intr_path = scene_dir / "camera" / cam_name / "intrinsics.json"
    return load_json(intr_path)


def load_camera_timestamps(scene_dir: Path, cam_name: str) -> list:
    """Load timestamps.json for a given camera."""
    ts_path = scene_dir / "camera" / cam_name / "timestamps.json"
    return load_json(ts_path)


def load_lidar_poses(scene_dir: Path) -> List[dict]:
    """Load lidar/poses.json."""
    return load_json(scene_dir / "lidar" / "poses.json")


def parse_intrinsics(intr_json: dict, img_w: int = 1920, img_h: int = 1080) -> np.ndarray:
    """
    Parse PandaSet intrinsics.json into a 3x3 camera intrinsic matrix K.

    PandaSet intrinsics.json format (typical):
        {
            "fx": ..., "fy": ...,
            "cx": ..., "cy": ...
        }
    Or:
        {
            "focal_length": ...,
            "principal_point": [cx, cy],
            "distortion": {...}
        }
    """
    K = np.eye(3, dtype=np.float64)

    if "fx" in intr_json and "fy" in intr_json:
        K[0, 0] = intr_json["fx"]
        K[1, 1] = intr_json["fy"]
        K[0, 2] = intr_json.get("cx", img_w / 2.0)
        K[1, 2] = intr_json.get("cy", img_h / 2.0)
    elif "focal_length" in intr_json:
        fl = intr_json["focal_length"]
        K[0, 0] = fl
        K[1, 1] = fl
        pp = intr_json.get("principal_point", [img_w / 2.0, img_h / 2.0])
        K[0, 2] = pp[0]
        K[1, 2] = pp[1]
    elif "intrinsics" in intr_json:
        # Some versions store as [fx, fy, cx, cy]
        vals = intr_json["intrinsics"]
        K[0, 0] = vals[0]
        K[1, 1] = vals[1]
        K[0, 2] = vals[2]
        K[1, 2] = vals[3]
    else:
        # Fallback: try common key patterns
        for key in ["f_x", "fx", "focal_x"]:
            if key in intr_json:
                K[0, 0] = intr_json[key]
                break
        for key in ["f_y", "fy", "focal_y"]:
            if key in intr_json:
                K[1, 1] = intr_json[key]
                break
        for key in ["c_x", "cx", "principal_x"]:
            if key in intr_json:
                K[0, 2] = intr_json[key]
                break
        for key in ["c_y", "cy", "principal_y"]:
            if key in intr_json:
                K[1, 2] = intr_json[key]
                break

    return K.astype(np.float32)


# ---------------------------------------------------------------------------
# Interpolation helpers (for 12Hz upsampling)
# ---------------------------------------------------------------------------
def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions [w,x,y,z]."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = math.acos(np.clip(dot, -1, 1))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


def interp_pose(pose0: dict, pose1: dict, t: float) -> dict:
    """
    Linearly interpolate between two PandaSet poses.
    pose format: {"position": {"x":..,"y":..,"z":..}, "heading": {"w":..,"x":..,"y":..,"z":..}}
    """
    p0, p1 = pose0["position"], pose1["position"]
    h0, h1 = pose0["heading"], pose1["heading"]
    pos = {
        "x": p0["x"] + t * (p1["x"] - p0["x"]),
        "y": p0["y"] + t * (p1["y"] - p0["y"]),
        "z": p0["z"] + t * (p1["z"] - p0["z"]),
    }
    q0 = np.array([h0["w"], h0["x"], h0["y"], h0["z"]])
    q1 = np.array([h1["w"], h1["x"], h1["y"], h1["z"]])
    qi = slerp(q0, q1, t)
    heading = {"w": qi[0], "x": qi[1], "y": qi[2], "z": qi[3]}
    return {"position": pos, "heading": heading}


def interp_K(K0: np.ndarray, K1: np.ndarray, t: float) -> np.ndarray:
    """Linearly interpolate two intrinsic matrices."""
    return (K0 * (1 - t) + K1 * t).astype(np.float32)


# ---------------------------------------------------------------------------
# Scene-level caption generation
# ---------------------------------------------------------------------------
def infer_weather_and_time(scene_dir: Path) -> dict:
    """
    Infer weather and time-of-day from scene metadata.
    PandaSet doesn't provide this directly, so we use defaults.
    Can be overridden by a manual annotation file.
    """
    caption = {"weather": "unknown", "time": "daytime"}
    # Try loading from meta if available
    meta_ts_path = scene_dir / "meta" / "timestamps.json"
    if meta_ts_path.exists():
        try:
            ts_data = load_json(meta_ts_path)
            if isinstance(ts_data, list) and len(ts_data) > 0:
                # Try to extract hour from timestamp
                first_ts = ts_data[0]
                if isinstance(first_ts, (int, float)):
                    # Unix timestamp in seconds or microseconds
                    import datetime
                    if first_ts > 1e12:
                        first_ts = first_ts / 1e6
                    dt = datetime.datetime.fromtimestamp(first_ts)
                    hour = dt.hour
                    if 6 <= hour < 18:
                        caption["time"] = "daytime"
                    elif 18 <= hour < 21:
                        caption["time"] = "dusk"
                    else:
                        caption["time"] = "night"
        except Exception:
            pass
    return caption


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------
def convert_scene(
    scene_dir: Path,
    data_dir: Path,
    interp_hz: int = 12,
    train_split: Optional[List[str]] = None,
    val_split: Optional[List[str]] = None,
) -> Tuple[List[dict], List[List[str]]]:
    """
    Convert a single PandaSet scene to the nuScenes-compatible info format.

    Args:
        scene_dir: Path to the scene directory (e.g., /data/pandaset/002)
        data_dir: Root path to PandaSet (for relative image paths)
        interp_hz: Target frame rate after interpolation (default 12Hz to match nuScenes)
        train_split: List of scene IDs for training (if filtering)
        val_split: List of scene IDs for validation (if filtering)

    Returns:
        infos: list of per-frame info dicts
        scene_tokens: list containing one list of tokens for this scene
    """
    scene_id = scene_dir.name
    print(f"  Converting scene {scene_id} ...")

    # Discover cameras
    cam_names = get_camera_names(scene_dir)
    if not cam_names:
        print(f"  WARNING: No cameras found in {scene_dir}, skipping.")
        return [], []

    # Map to nuScenes view names and load per-camera data
    cam_data = {}
    for cam_name in cam_names:
        nusc_name = PANDASET_TO_NUSCENES.get(cam_name)
        if nusc_name is None:
            print(f"  WARNING: Unknown camera {cam_name}, skipping.")
            continue
        poses = load_camera_poses(scene_dir, cam_name)
        intrinsics_json = load_camera_intrinsics(scene_dir, cam_name)
        K = parse_intrinsics(intrinsics_json)
        cam_data[nusc_name] = {
            "pandaset_name": cam_name,
            "poses": poses,
            "K": K,
            "intrinsics_json": intrinsics_json,
        }

    if not cam_data:
        print(f"  WARNING: No valid cameras in scene {scene_id}, skipping.")
        return [], []

    # Load LiDAR poses as ego poses (LiDAR is the reference frame)
    lidar_poses = load_lidar_poses(scene_dir)
    num_raw_frames = len(lidar_poses)
    print(f"    Raw frames: {num_raw_frames}, cameras: {list(cam_data.keys())}")

    # In PandaSet, LiDAR runs at 10Hz. Determine original FPS.
    original_hz = 10  # PandaSet default

    # Compute interpolation factor
    if interp_hz > original_hz:
        interp_factor = interp_hz / original_hz
    else:
        interp_factor = 1.0

    # Generate frame indices with interpolation
    interp_frames = []
    for i in range(num_raw_frames - 1):
        num_interp = int(interp_factor)
        for j in range(num_interp):
            t = j / num_interp
            interp_frames.append((i, i + 1, t))
    # Always include the last frame
    interp_frames.append((num_raw_frames - 1, num_raw_frames - 1, 0.0))

    # Get scene-level caption
    caption = infer_weather_and_time(scene_dir)

    # Build infos
    infos = []
    scene_tokens = []
    for idx, (frame_i, frame_j, t_frac) in enumerate(interp_frames):
        token = f"{scene_id}_{idx:06d}"

        # Interpolate LiDAR (ego) pose
        if t_frac == 0.0:
            lidar_pose = lidar_poses[frame_i]
        else:
            lidar_pose = interp_pose(lidar_poses[frame_i], lidar_poses[frame_j], t_frac)

        lidar_T = pose_to_transform(lidar_pose)  # lidar→world
        lidar_R = lidar_T[:3, :3]
        lidar_t = lidar_T[:3, 3]
        lidar2ego_q = rotation_matrix_to_quaternion(lidar_R)

        # In PandaSet, lidar frame ≈ ego frame (vehicle body),
        # so lidar2ego is identity (or a small fixed offset)
        # We set lidar2ego as identity for simplicity
        lidar2ego_translation = [0.0, 0.0, 0.0]
        lidar2ego_rotation = [1.0, 0.0, 0.0, 0.0]  # identity quaternion

        # ego2global = lidar2global since lidar≈ego
        ego2global_translation = lidar_t.tolist()
        ego2global_rotation = lidar2ego_q.tolist()

        # Build per-camera info
        cams = {}
        for nusc_name, cd in cam_data.items():
            cam_poses = cd["poses"]
            K = cd["K"]

            # Interpolate camera pose
            if t_frac == 0.0:
                cam_pose = cam_poses[frame_i]
            else:
                cam_pose = interp_pose(cam_poses[frame_i], cam_poses[frame_j], t_frac)

            cam_T = pose_to_transform(cam_pose)  # cam→world

            # sensor2ego: camera→ego (= camera→world * world→ego)
            # Since ego≈lidar, we compute cam→lidar = inv(lidar_T) @ cam_T
            cam2lidar_T = np.linalg.inv(lidar_T) @ cam_T
            sensor2ego_R = cam2lidar_T[:3, :3]
            sensor2ego_t = cam2lidar_T[:3, 3]
            sensor2ego_q = rotation_matrix_to_quaternion(sensor2ego_R)

            # sensor2lidar_rotation and translation (as used in _get_camera_params)
            sensor2lidar_rotation = sensor2ego_R.astype(np.float32)
            sensor2lidar_translation = sensor2ego_t.astype(np.float32)

            # Image path (relative to data_dir)
            pandaset_name = cd["pandaset_name"]
            img_filename = f"{frame_i:02d}.jpg"
            rel_img_path = str(data_dir / scene_id / "camera" / pandaset_name / img_filename)

            # For interpolated frames, the image is from the nearest key frame
            if t_frac > 0.0 and frame_j < len(cam_poses):
                # Use the closer frame's image
                nearest_idx = frame_i if t_frac < 0.5 else frame_j
                img_filename = f"{nearest_idx:02d}.jpg"
                rel_img_path = str(data_dir / scene_id / "camera" / pandaset_name / img_filename)

            # Camera's own ego2global (same as lidar's for this frame)
            cam_ego2global_translation = ego2global_translation
            cam_ego2global_rotation = ego2global_rotation

            cams[nusc_name] = {
                "data_path": rel_img_path,
                "type": nusc_name,
                "sample_data_token": f"{token}_{nusc_name}",
                "sensor2ego_translation": sensor2ego_t.tolist(),
                "sensor2ego_rotation": sensor2ego_q.tolist(),  # [w,x,y,z]
                "ego2global_translation": cam_ego2global_translation,
                "ego2global_rotation": cam_ego2global_rotation,
                "timestamp": float(idx) / interp_hz,
                "sensor2lidar_rotation": sensor2lidar_rotation,
                "sensor2lidar_translation": sensor2lidar_translation,
                "camera_intrinsics": K,
            }

        # LiDAR path (for reference)
        lidar_filename = f"{frame_i:02d}.pkl.gz"
        lidar_path = str(data_dir / scene_id / "lidar" / lidar_filename)

        # Timestamp
        timestamp = float(idx) / interp_hz

        info = {
            "token": token,
            "timestamp": timestamp,
            "location": f"pandaset_{scene_id}",
            "timeofday": caption.get("time", "daytime"),
            "is_key_frame": (t_frac == 0.0),
            "caption": caption,
            "lidar_path": lidar_path,
            "cams": cams,
            "lidar2ego_translation": lidar2ego_translation,
            "lidar2ego_rotation": lidar2ego_rotation,
            "ego2global_translation": ego2global_translation,
            "ego2global_rotation": ego2global_rotation,
            # Placeholder fields required by the dataset class
            "gt_boxes": np.zeros((0, 7), dtype=np.float32),
            "gt_names": [],
            "gt_velocity": [],
            "num_lidar_pts": [],
            "num_radar_pts": [],
            "valid_flag": [],
            "gt_box_ids": [],
            "sweeps": [],
        }

        infos.append(info)
        scene_tokens.append(token)

    return infos, [scene_tokens]


def convert_pandaset(
    data_dir: str,
    output_dir: str,
    interp_hz: int = 12,
    train_ratio: float = 0.8,
    val_scene_ids: Optional[List[str]] = None,
):
    """
    Convert entire PandaSet dataset to nuScenes-compatible PKL files.

    Args:
        data_dir: Root directory of PandaSet (containing scene folders)
        output_dir: Output directory for PKL files
        interp_hz: Target frame rate for interpolation (default 12Hz)
        train_ratio: Ratio of scenes for training if val_scene_ids not specified
        val_scene_ids: Explicit list of scene IDs for validation split
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Discover all scenes
    scene_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and (d / "camera").exists()
    ])

    if not scene_dirs:
        print(f"ERROR: No PandaSet scenes found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(scene_dirs)} scenes in {data_dir}")

    # Determine train/val split
    scene_ids = [d.name for d in scene_dirs]

    # Filter scenes with semantic segmentation if needed
    semseg_scenes = []
    for sd in scene_dirs:
        semseg_dir = sd / "annotations" / "semseg"
        if semseg_dir.exists() and list(semseg_dir.glob("*.pkl.gz")):
            semseg_scenes.append(sd.name)
    print(f"  Scenes with semantic segmentation: {len(semseg_scenes)}")

    if val_scene_ids is not None:
        train_ids = [s for s in scene_ids if s not in val_scene_ids]
        val_ids = val_scene_ids
    else:
        # Use official-like split: ~80/23
        # PandaSet has 103 scenes total, 76 with semseg
        # We split by scene ID ordering
        n_val = max(1, int(len(scene_ids) * (1 - train_ratio)))
        val_ids = scene_ids[-n_val:]
        train_ids = scene_ids[:-n_val]

    print(f"  Train scenes: {len(train_ids)}, Val scenes: {len(val_ids)}")

    # Convert train split
    print("\n=== Converting TRAIN split ===")
    train_infos = []
    train_scene_tokens = []
    for sd in scene_dirs:
        if sd.name not in train_ids:
            continue
        infos, scene_tokens = convert_scene(sd, data_path, interp_hz)
        train_infos.extend(infos)
        train_scene_tokens.extend(scene_tokens)

    train_pkl = {
        "infos": train_infos,
        "metadata": {"version": f"pandaset_interp_{interp_hz}Hz_train"},
        "scene_tokens": train_scene_tokens,
    }
    train_pkl_path = output_path / f"pandaset_interp_{interp_hz}Hz_infos_train.pkl"
    print(f"  Saving train PKL to {train_pkl_path} ({len(train_infos)} frames, {len(train_scene_tokens)} scenes)")
    with open(train_pkl_path, "wb") as f:
        pickle.dump(train_pkl, f)

    # Convert val split
    print("\n=== Converting VAL split ===")
    val_infos = []
    val_scene_tokens = []
    for sd in scene_dirs:
        if sd.name not in val_ids:
            continue
        infos, scene_tokens = convert_scene(sd, data_path, interp_hz)
        val_infos.extend(infos)
        val_scene_tokens.extend(scene_tokens)

    val_pkl = {
        "infos": val_infos,
        "metadata": {"version": f"pandaset_interp_{interp_hz}Hz_val"},
        "scene_tokens": val_scene_tokens,
    }
    val_pkl_path = output_path / f"pandaset_interp_{interp_hz}Hz_infos_val.pkl"
    print(f"  Saving val PKL to {val_pkl_path} ({len(val_infos)} frames, {len(val_scene_tokens)} scenes)")
    with open(val_pkl_path, "wb") as f:
        pickle.dump(val_pkl, f)

    # Save split info
    split_info = {
        "train_scene_ids": train_ids,
        "val_scene_ids": val_ids,
        "semseg_scene_ids": semseg_scenes,
        "interp_hz": interp_hz,
        "total_frames_train": len(train_infos),
        "total_frames_val": len(val_infos),
        "total_scenes_train": len(train_scene_tokens),
        "total_scenes_val": len(val_scene_tokens),
    }
    split_path = output_path / "split_info.json"
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\n  Split info saved to {split_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"  Total scenes:       {len(scene_ids)}")
    print(f"  Train scenes:       {len(train_ids)} ({len(train_infos)} frames)")
    print(f"  Val scenes:         {len(val_ids)} ({len(val_infos)} frames)")
    print(f"  Interpolation:      {interp_hz}Hz")
    print(f"  Output directory:   {output_path}")
    print(f"  Train PKL:          {train_pkl_path}")
    print(f"  Val PKL:            {val_pkl_path}")
    print("=" * 60)

    # Verify compatibility
    verify_pkl(train_pkl_path)


def verify_pkl(pkl_path: Path):
    """Verify that the generated PKL is compatible with NuscenesVideoDataset."""
    print(f"\nVerifying {pkl_path} ...")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Check top-level keys
    required_keys = {"infos", "metadata", "scene_tokens"}
    assert required_keys.issubset(set(data.keys())), \
        f"Missing top-level keys: {required_keys - set(data.keys())}"

    # Check first info
    if data["infos"]:
        info = data["infos"][0]
        required_info_keys = {
            "token", "timestamp", "cams", "lidar2ego_translation",
            "lidar2ego_rotation", "ego2global_translation", "ego2global_rotation"
        }
        missing = required_info_keys - set(info.keys())
        assert not missing, f"Missing info keys: {missing}"

        # Check cam structure
        for cam_name, cam_info in info["cams"].items():
            required_cam_keys = {
                "data_path", "sensor2ego_translation", "sensor2ego_rotation",
                "ego2global_translation", "ego2global_rotation",
                "sensor2lidar_rotation", "sensor2lidar_translation",
                "camera_intrinsics", "timestamp"
            }
            missing_cam = required_cam_keys - set(cam_info.keys())
            assert not missing_cam, f"Missing cam keys for {cam_name}: {missing_cam}"

            # Check types
            assert isinstance(cam_info["sensor2lidar_rotation"], np.ndarray), \
                f"sensor2lidar_rotation should be ndarray, got {type(cam_info['sensor2lidar_rotation'])}"
            assert cam_info["sensor2lidar_rotation"].shape == (3, 3), \
                f"sensor2lidar_rotation shape should be (3,3), got {cam_info['sensor2lidar_rotation'].shape}"
            assert isinstance(cam_info["camera_intrinsics"], np.ndarray), \
                f"camera_intrinsics should be ndarray, got {type(cam_info['camera_intrinsics'])}"
            assert cam_info["camera_intrinsics"].shape == (3, 3), \
                f"camera_intrinsics shape should be (3,3), got {cam_info['camera_intrinsics'].shape}"

    # Check scene_tokens
    assert len(data["scene_tokens"]) > 0, "scene_tokens is empty"
    total_tokens_from_scenes = sum(len(s) for s in data["scene_tokens"])
    assert total_tokens_from_scenes == len(data["infos"]), \
        f"Mismatch: {total_tokens_from_scenes} tokens in scenes vs {len(data['infos'])} infos"

    print("  Verification PASSED!")
    print(f"    infos: {len(data['infos'])}")
    print(f"    scenes: {len(data['scene_tokens'])}")
    print(f"    cam views: {list(data['infos'][0]['cams'].keys())}")
    print(f"    sample token: {data['infos'][0]['token']}")
    print(f"    sample cam path: {data['infos'][0]['cams']['CAM_FRONT']['data_path']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert PandaSet to nuScenes-compatible PKL")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory of PandaSet (containing scene folders like 002/, 004/, ...)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/pandaset_pkl",
        help="Output directory for PKL files"
    )
    parser.add_argument(
        "--interp_hz", type=int, default=12,
        help="Target frame rate after interpolation (default: 12Hz to match nuScenes pipeline)"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Ratio of scenes for training (default: 0.8)"
    )
    parser.add_argument(
        "--val_scene_ids", type=str, nargs="+", default=None,
        help="Explicit list of scene IDs for validation split (overrides train_ratio)"
    )
    args = parser.parse_args()

    convert_pandaset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        interp_hz=args.interp_hz,
        train_ratio=args.train_ratio,
        val_scene_ids=args.val_scene_ids,
    )


if __name__ == "__main__":
    main()
