import os
import cv2
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torchvision.models import inception_v3
from torchvision import transforms


SEG_COLOR_MAP = [
    (0, 0, 0),
    (0, 207, 191),
    (135, 206, 235),
    (255, 158, 0),
    (0, 0, 230),
    (255, 61, 99),
    (192, 192, 192),
    (75, 0, 75),
    (0, 175, 0),
]

SEG_CLASS_NAMES = [
    "object", "road", "sky", "car", "person",
    "bicycle", "lane", "sidewalk", "vegetation",
]


class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        inception = inception_v3(weights='DEFAULT')
        inception.eval()
        inception.to(device)
        self.device = device
        self.inception = inception
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    @torch.no_grad()
    def extract_features(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        B = images.shape[0]
        if images.shape[-1] != 299 or images.shape[-2] != 299:
            images = torch.nn.functional.interpolate(
                images, size=(299, 299), mode='bilinear', align_corners=False
            )
        images = self.transform(images)
        x = self.inception.Conv2d_1a_3x3(images.to(self.device))
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        features = x.view(B, -1).cpu()
        return features

    @torch.no_grad()
    def extract_features_from_video(self, video_tensor):
        if video_tensor.dim() == 4:
            C, T, H, W = video_tensor.shape
            frames = []
            for t in range(T):
                frame = video_tensor[:, t]
                feat = self.extract_features(frame.unsqueeze(0))
                frames.append(feat)
            return torch.cat(frames, dim=0)
        elif video_tensor.dim() == 5:
            N, C, T, H, W = video_tensor.shape
            all_feats = []
            for n in range(N):
                frames = []
                for t in range(T):
                    frame = video_tensor[n, :, t]
                    feat = self.extract_features(frame.unsqueeze(0))
                    frames.append(feat)
                all_feats.append(torch.cat(frames, dim=0))
            return all_feats
        return None


def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)


def compute_fid(pred_features, gt_features):
    pred_np = pred_features.numpy()
    gt_np = gt_features.numpy()
    mu_pred = np.mean(pred_np, axis=0)
    sigma_pred = np.cov(pred_np, rowvar=False)
    mu_gt = np.mean(gt_np, axis=0)
    sigma_gt = np.cov(gt_np, rowvar=False)
    return compute_frechet_distance(mu_pred, sigma_pred, mu_gt, sigma_gt)


def compute_fvd(pred_video_features, gt_video_features):
    if isinstance(pred_video_features, list):
        pred_concat = torch.cat(pred_video_features, dim=0)
        gt_concat = torch.cat(gt_video_features, dim=0)
    else:
        pred_concat = pred_video_features
        gt_concat = gt_video_features
    pred_np = pred_concat.numpy()
    gt_np = gt_concat.numpy()
    mu_pred = np.mean(pred_np, axis=0)
    sigma_pred = np.cov(pred_np, rowvar=False)
    mu_gt = np.mean(gt_np, axis=0)
    sigma_gt = np.cov(gt_np, rowvar=False)
    return compute_frechet_distance(mu_pred, sigma_pred, mu_gt, sigma_gt)


def compute_depth_metrics(pred, gt, valid_mask=None, align_scale=True):
    if valid_mask is not None:
        pred = pred[valid_mask]
        gt = gt[valid_mask]
    else:
        pred = pred.flatten().float()
        gt = gt.flatten().float()

    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]

    if len(gt) == 0:
        return None

    # Scale-invariant alignment via median scaling (Eigen et al., 2014)
    if align_scale:
        median_gt = torch.median(gt)
        median_pred = torch.median(pred)
        if median_pred > 1e-6:
            scale = median_gt / median_pred
            pred = pred * scale

    abs_rel = (torch.abs(pred - gt) / gt).mean().item()
    sq_rel = ((pred - gt) ** 2 / gt).mean().item()
    rmse = torch.sqrt(((pred - gt) ** 2).mean()).item()
    rmse_log = torch.sqrt((torch.log(pred.clamp(min=1e-6)) - torch.log(gt.clamp(min=1e-6))) ** 2).mean().item()

    ratio = torch.max(pred / gt.clamp(min=1e-6), gt / pred.clamp(min=1e-6))
    delta_125 = (ratio < 1.25).float().mean().item()
    delta_1252 = (ratio < 1.25 ** 2).float().mean().item()
    delta_1253 = (ratio < 1.25 ** 3).float().mean().item()

    return {
        'Abs Rel': abs_rel,
        'Sq Rel': sq_rel,
        'RMSE': rmse,
        'RMSE log': rmse_log,
        'delta<1.25': delta_125,
        'delta<1.25^2': delta_1252,
        'delta<1.25^3': delta_1253,
    }


def compute_miou(pred, gt, num_classes=9, ignore_index=-1):
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        gt_cls = (gt == cls)
        if ignore_index >= 0:
            valid = (gt != ignore_index)
            pred_cls = pred_cls & valid
            gt_cls = gt_cls & valid
        intersection = (pred_cls & gt_cls).sum().float()
        union = (pred_cls | gt_cls).sum().float()
        if union > 0:
            iou_per_class.append((intersection / union).item())
    if len(iou_per_class) == 0:
        return 0.0, {}
    miou = np.mean(iou_per_class)
    class_iou = {}
    for i, cls in enumerate(range(num_classes)):
        if i < len(iou_per_class):
            class_iou[SEG_CLASS_NAMES[cls]] = iou_per_class[i]
        else:
            class_iou[SEG_CLASS_NAMES[cls]] = 0.0
    return miou, class_iou


DEPTH_BASE_DIRS = [
    'data/nuscenes_12hz_depth_unzip',
    'dist4d/depth_process/nus_Rdepth/release_data_nus_Rdepth/11_scene',
    '../dist4d/depth_process/nus_Rdepth/release_data_nus_Rdepth/11_scene',
]

PKL_PATHS = [
    'dist4d/data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val.pkl',
    'dist4d/data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train.pkl',
    '../dist4d/data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val.pkl',
    '../dist4d/data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train.pkl',
]


def build_path_to_token_map(pkl_paths=None):
    path_to_token = {}
    if pkl_paths is None:
        pkl_paths = PKL_PATHS
    for pkl_path in pkl_paths:
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        for info in data['infos']:
            sample_token = info['token']
            for cam_name, cam_info in info['cams'].items():
                data_path = cam_info['data_path']
                filename = os.path.basename(data_path)
                path_to_token[filename] = sample_token
    return path_to_token


def find_depth_base_dir():
    for d in DEPTH_BASE_DIRS:
        if os.path.isdir(d):
            return d
    return None


def load_gt_depth(token, view_name, target_h, target_w, max_depth=100.0, depth_base=None):
    if depth_base is None:
        depth_base = find_depth_base_dir()
    if depth_base is None:
        return None
    depth_path = Path(depth_base) / token / view_name / "refined_depth.npz"
    if not depth_path.exists():
        return None
    depth_map_ori = np.load(depth_path)['depth_pred']
    depth_map_resize = cv2.resize(depth_map_ori, (target_w, target_h))
    depth_map_resize[depth_map_resize > max_depth] = max_depth
    return depth_map_resize


def load_gt_seg(image_path, target_h, target_w):
    seg_class_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 0, 7: 6, 8: 7, 9: 8}
    seg_path = image_path.replace('/nuscenes/', '/nuscenes_seg/').replace('.jpg', '.png') \
                         .replace('/samples/', '/samples_seg/').replace('/sweeps/', '/sweeps_seg/')
    if not os.path.exists(seg_path):
        return None
    seg_mask_ori = cv2.imread(seg_path)
    if seg_mask_ori is None:
        return None
    seg_mask_resize = cv2.resize(seg_mask_ori, (target_w, target_h), interpolation=cv2.INTER_NEAREST)[:, :, 0]
    seg_label = np.zeros_like(seg_mask_resize)
    for seg_id in np.unique(seg_mask_resize):
        new_id = seg_class_map.get(int(seg_id), 0)
        seg_label[seg_mask_resize == seg_id] = new_id
    return seg_label


def extract_rgb_features(pred_rgb, gt_rgb, feature_extractor):
    pred_rgb = pred_rgb.float().clamp(0, 1)
    gt_rgb = gt_rgb.float().clamp(0, 1)

    pred_frame_feats = []
    gt_frame_feats = []
    pred_video_feats = []
    gt_video_feats = []

    if pred_rgb.dim() == 5:
        N, C, T, H, W = pred_rgb.shape
        for n in range(N):
            p_video = pred_rgb[n]
            g_video = gt_rgb[n]
            if p_video.shape != g_video.shape:
                continue
            p_feats = feature_extractor.extract_features_from_video(p_video)
            g_feats = feature_extractor.extract_features_from_video(g_video)
            if p_feats is not None:
                if isinstance(p_feats, torch.Tensor):
                    pred_frame_feats.append(p_feats)
                    gt_frame_feats.append(g_feats)
                    pred_video_feats.append(p_feats.mean(dim=0, keepdim=True))
                    gt_video_feats.append(g_feats.mean(dim=0, keepdim=True))
                elif isinstance(p_feats, list):
                    for pf, gf in zip(p_feats, g_feats):
                        pred_frame_feats.append(pf)
                        gt_frame_feats.append(gf)
                        pred_video_feats.append(pf.mean(dim=0, keepdim=True))
                        gt_video_feats.append(gf.mean(dim=0, keepdim=True))
    elif pred_rgb.dim() == 4:
        C, T, H, W = pred_rgb.shape
        p_feats = feature_extractor.extract_features_from_video(pred_rgb)
        g_feats = feature_extractor.extract_features_from_video(gt_rgb)
        if p_feats is not None:
            pred_frame_feats.append(p_feats)
            gt_frame_feats.append(g_feats)
            pred_video_feats.append(p_feats.mean(dim=0, keepdim=True))
            gt_video_feats.append(g_feats.mean(dim=0, keepdim=True))

    return pred_frame_feats, gt_frame_feats, pred_video_feats, gt_video_feats


def _resolve_path(p):
    while isinstance(p, list):
        if len(p) == 0:
            return None
        p = p[0]
    return p


def _resolve_token(t):
    while isinstance(t, list):
        if len(t) == 0:
            return None
        t = t[0]
    if isinstance(t, str):
        return t
    return None


def evaluate_depth_from_tensors(pred_depth, gt_depth_tensors, paths, max_depth=100.0,
                                tokens=None, path_to_token=None, depth_base=None):
    all_metrics = defaultdict(list)
    if pred_depth.dim() == 5:
        N, _, T, H, W = pred_depth.shape
    elif pred_depth.dim() == 4:
        _, T, H, W = pred_depth.shape
        N = 1
        pred_depth = pred_depth.unsqueeze(0)
    else:
        return None

    view_order = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                  "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]

    # N is the number of views (6 for multi-view) or 1 for single-view.
    # When N equals the number of views, each index n corresponds to view_order[n].
    # When N == 1, we iterate all views and compare against the single prediction.
    num_views = len(view_order)
    is_multiview = (N == num_views)

    for t in range(T):
        # Resolve token for this frame
        token = None
        if tokens is not None and t < len(tokens):
            token = _resolve_token(tokens[t])

        if token is None:
            frame_paths = paths[min(t, len(paths) - 1)]
            img_path = None
            for v_idx, view_name in enumerate(view_order):
                if v_idx < len(frame_paths):
                    img_path = _resolve_path(frame_paths[v_idx])
                    if img_path is not None:
                        break
            if img_path is None:
                continue
            if path_to_token is not None:
                filename = os.path.basename(img_path)
                token = path_to_token.get(filename)
                if token is None:
                    continue
            else:
                token = img_path.split('/')[-3]

        if token is None:
            continue

        # Iterate over views
        for v_idx, view_name in enumerate(view_order):
            gt_depth = load_gt_depth(token, view_name, H, W, max_depth, depth_base)
            if gt_depth is None:
                continue

            # Match prediction to the correct view
            if is_multiview:
                if v_idx >= N:
                    continue
                pred_d = pred_depth[v_idx, 0, t].numpy()
            else:
                pred_d = pred_depth[0, 0, t].numpy()

            valid = (gt_depth > 0) & (gt_depth < max_depth)
            metrics = compute_depth_metrics(
                torch.from_numpy(pred_d), torch.from_numpy(gt_depth), valid
            )
            if metrics is not None:
                for k, v in metrics.items():
                    all_metrics[k].append(v)

    if len(all_metrics) == 0:
        return None
    return {k: np.mean(v) for k, v in all_metrics.items()}


def evaluate_seg_from_tensors(pred_seg, paths, num_classes=9):
    all_miou = []
    all_class_iou = defaultdict(list)

    if pred_seg.dim() == 5:
        N, _, T, H, W = pred_seg.shape
    elif pred_seg.dim() == 4:
        _, T, H, W = pred_seg.shape
        N = 1
        pred_seg = pred_seg.unsqueeze(0)
    else:
        return None

    view_order = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                  "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]

    for n in range(N):
        for t in range(T):
            frame_paths = paths[min(t, len(paths) - 1)]
            for v_idx, view_name in enumerate(view_order):
                if v_idx >= len(frame_paths):
                    continue
                img_path = _resolve_path(frame_paths[v_idx])
                if img_path is None:
                    continue
                gt_seg = load_gt_seg(img_path, H, W)
                if gt_seg is None:
                    continue
                pred_s = pred_seg[n, 0, t].numpy().astype(np.int64)
                miou, class_iou = compute_miou(
                    torch.from_numpy(pred_s.flatten()),
                    torch.from_numpy(gt_seg.flatten()),
                    num_classes=num_classes,
                )
                all_miou.append(miou)
                for cls_name, iou_val in class_iou.items():
                    all_class_iou[cls_name].append(iou_val)

    if len(all_miou) == 0:
        return None
    result = {'mIoU': np.mean(all_miou)}
    for cls_name, iou_list in all_class_iou.items():
        result[f'IoU_{cls_name}'] = np.mean(iou_list)
    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate OmniNWM outputs')
    parser.add_argument('--eval_dir', type=str, required=True,
                        help='Path to eval_tensors directory')
    parser.add_argument('--eval_rgb', action='store_true', help='Evaluate RGB quality (FID/FVD)')
    parser.add_argument('--eval_depth', action='store_true', help='Evaluate Depth quality')
    parser.add_argument('--eval_seg', action='store_true', help='Evaluate Seg quality')
    parser.add_argument('--eval_all', action='store_true', help='Evaluate all modalities')
    parser.add_argument('--max_depth', type=float, default=100.0)
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for feature extraction (cuda/cpu)')
    args = parser.parse_args()

    if args.eval_all:
        args.eval_rgb = True
        args.eval_depth = True
        args.eval_seg = True

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Error: {eval_dir} does not exist")
        return

    rgb_files = sorted([f for f in eval_dir.glob("*_rgb.pt") if "_gt_rgb" not in f.name])
    print(f"Found {len(rgb_files)} samples in {eval_dir}")

    if len(rgb_files) == 0:
        print("No eval tensor files found. Did you set save_for_eval=True in config?")
        return

    feature_extractor = None
    if args.eval_rgb:
        device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
        print(f"Loading InceptionV3 feature extractor on {device}...")
        feature_extractor = InceptionFeatureExtractor(device=device)

    path_to_token = None
    depth_base = None
    if args.eval_depth:
        print("Building path-to-token mapping from pkl files...")
        path_to_token = build_path_to_token_map()
        print(f"  Mapped {len(path_to_token)} image paths to tokens")
        depth_base = find_depth_base_dir()
        if depth_base is not None:
            print(f"  Found depth data at: {depth_base}")
        else:
            print("  Warning: No depth data directory found!")

    all_depth_metrics = defaultdict(list)
    all_seg_metrics = defaultdict(list)

    scene_pred_frame_feats = []
    scene_gt_frame_feats = []
    scene_pred_video_feats = []
    scene_gt_video_feats = []

    for rgb_file in tqdm(rgb_files, desc="Evaluating"):
        stem = rgb_file.stem.replace("_rgb", "")
        prefix = stem

        if args.eval_rgb:
            pred_rgb = torch.load(rgb_file, map_location='cpu')
            gt_file = eval_dir / f"{prefix}_gt_rgb.pt"
            if gt_file.exists():
                gt_rgb = torch.load(gt_file, map_location='cpu')
                min_t = min(pred_rgb.shape[2], gt_rgb.shape[2])
                pred_rgb = pred_rgb[:, :, :min_t]
                gt_rgb = gt_rgb[:, :, :min_t]
                pf, gf, pv, gv = extract_rgb_features(pred_rgb, gt_rgb, feature_extractor)
                if len(pf) > 0:
                    scene_pred_frame_feats.append(torch.cat(pf, dim=0))
                    scene_gt_frame_feats.append(torch.cat(gf, dim=0))
                if len(pv) > 0:
                    scene_pred_video_feats.append(torch.cat(pv, dim=0))
                    scene_gt_video_feats.append(torch.cat(gv, dim=0))
            else:
                print(f"Warning: GT RGB not found for {prefix}")

        if args.eval_depth:
            depth_file = eval_dir / f"{prefix}_depth.pt"
            paths_file = eval_dir / f"{prefix}_paths.pt"
            tokens_file = eval_dir / f"{prefix}_tokens.pt"
            if depth_file.exists() and paths_file.exists():
                pred_depth = torch.load(depth_file, map_location='cpu')
                paths = torch.load(paths_file, map_location='cpu')
                tokens = None
                if tokens_file.exists():
                    tokens = torch.load(tokens_file, map_location='cpu')
                metrics = evaluate_depth_from_tensors(
                    pred_depth, None, paths, max_depth=args.max_depth,
                    tokens=tokens, path_to_token=path_to_token, depth_base=depth_base
                )
                if metrics is not None:
                    for k, v in metrics.items():
                        all_depth_metrics[k].append(v)
            else:
                print(f"Warning: Depth or paths not found for {prefix}")

        if args.eval_seg:
            seg_file = eval_dir / f"{prefix}_seg.pt"
            paths_file = eval_dir / f"{prefix}_paths.pt"
            if seg_file.exists() and paths_file.exists():
                pred_seg = torch.load(seg_file, map_location='cpu')
                paths = torch.load(paths_file, map_location='cpu')
                metrics = evaluate_seg_from_tensors(
                    pred_seg, paths, num_classes=args.num_classes
                )
                if metrics is not None:
                    for k, v in metrics.items():
                        all_seg_metrics[k].append(v)
            else:
                print(f"Warning: Seg or paths not found for {prefix}")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    if args.eval_rgb and len(scene_pred_frame_feats) > 0:
        print("\n--- RGB Quality ---")
        total_pred_frames = sum(f.shape[0] for f in scene_pred_frame_feats)
        total_gt_frames = sum(f.shape[0] for f in scene_gt_frame_feats)
        print(f"  Total scenes: {len(scene_pred_frame_feats)}")
        print(f"  Total frames - Pred: {total_pred_frames}, GT: {total_gt_frames}")

        fid_scores = []
        for i, (pred_feats, gt_feats) in enumerate(zip(scene_pred_frame_feats, scene_gt_frame_feats)):
            if pred_feats.shape[0] >= 2 and gt_feats.shape[0] >= 2:
                fid = compute_fid(pred_feats, gt_feats)
                fid_scores.append(fid)

        if len(fid_scores) > 0:
            print(f"  FID (per-scene avg): {np.mean(fid_scores):.4f}")
            print(f"  FID (per-scene std): {np.std(fid_scores):.4f}")
            print(f"  FID (per-scene min): {np.min(fid_scores):.4f}")
            print(f"  FID (per-scene max): {np.max(fid_scores):.4f}")
        else:
            print("  FID: Not enough frames per scene for computation (need >= 2)")

        all_pred_video = torch.cat(scene_pred_video_feats, dim=0)
        all_gt_video = torch.cat(scene_gt_video_feats, dim=0)
        print(f"  FVD video features - Pred: {all_pred_video.shape[0]}, GT: {all_gt_video.shape[0]}")
        if all_pred_video.shape[0] >= 2 and all_gt_video.shape[0] >= 2:
            fvd_score = compute_fvd(all_pred_video, all_gt_video)
            print(f"  FVD: {fvd_score:.4f}")
        else:
            print("  FVD: Not enough video features for computation")

    if args.eval_depth and len(all_depth_metrics) > 0:
        print("\n--- Depth Quality ---")
        for k, v_list in all_depth_metrics.items():
            print(f"  {k}: {np.mean(v_list):.4f}")
    elif args.eval_depth:
        print("\n--- Depth Quality ---")
        print("  No depth metrics computed!")
        if depth_base is None:
            print("  Reason: Depth GT data directory not found")
        else:
            print(f"  Reason: No matching GT depth found for any eval sample")
            print(f"  Depth data dir: {depth_base} ({len(os.listdir(depth_base))} entries)")
            print(f"  Check if eval samples belong to scenes with depth GT")

    if args.eval_seg and len(all_seg_metrics) > 0:
        print("\n--- Segmentation Quality ---")
        for k, v_list in all_seg_metrics.items():
            print(f"  {k}: {np.mean(v_list):.4f}")
    elif args.eval_seg:
        print("\n--- Segmentation Quality ---")
        print("  No segmentation metrics computed!")
        print("  Reason: Segmentation GT data (nuscenes_seg) not found on disk")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
