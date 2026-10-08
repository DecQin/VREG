"""Per-frame SSIM distance curves across checkpoints (v2).

横轴 = 帧号(0..12)；纵轴 = 1 - SSIM(gen vs GT)（距离，越小越好）。
一条线 = 一个 checkpoint。

SSIM 比像素 RMSE 更敏感于感知退化（模糊、结构扭曲、物体重影），
与你之前观察到的"只能保证第一帧"在视觉语义上更对齐。

依赖：scikit-image（已安装）
用法：python tools/per_frame_fid.py
"""

import glob
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

EVAL_DIRS = {
    # step5k/10k的eval目录只保存了txt分数，没存pt张量，无法画per-frame
    'P6 step20k':  '/data1/qjyuan/OmniNWM/outputs/ablation_pandaset/p6_anchor_cache_4gpu_eval_step20000/video/eval_tensors',
    'Baseline':    '/data1/qjyuan/OmniNWM/outputs/infer_baseline/video/eval_tensors',
    # 'P7 step6k':  '/data1/qjyuan/OmniNWM/outputs/ablation_pandaset/p7_anchor_cache_multiframe_4gpu_eval_step6000/video/eval_tensors',
}

OUT_PNG = 'tools/fig2_per_frame_ssim.png'
T_FRAMES = 13


def per_view_ssim_distance(gen_v, gt_v):
    """单帧 gen/gt: [V, C, H, W] -> mean 1-SSIM over views."""
    dists = []
    for v in range(gen_v.shape[0]):
        # skimage SSIM要灰度或多通道；对彩色直接给data_range
        # GT/gen范围在[0,1]（即使有bug也保持同一尺度）
        s = ssim(
            gen_v[v].transpose(1, 2, 0),
            gt_v[v].transpose(1, 2, 0),
            data_range=1.0,
            channel_axis=2,
            win_size=7,
        )
        dists.append(1.0 - s)
    return float(np.mean(dists))


def per_frame_curve(eval_dir):
    files = sorted(glob.glob(os.path.join(eval_dir, '[0-9]*_rgb.pt')))
    files = [f for f in files if '_gt_' not in f]
    by_frame = [[] for _ in range(T_FRAMES)]
    for gf in files:
        gtf = gf.replace('_rgb.pt', '_gt_rgb.pt')
        if not os.path.exists(gtf):
            continue
        gen = torch.load(gf, map_location='cpu', weights_only=True).float().numpy()
        gt = torch.load(gtf, map_location='cpu', weights_only=True).float().numpy()
        if gen.shape[2] < T_FRAMES:
            continue
        for t in range(T_FRAMES):
            by_frame[t].append(per_view_ssim_distance(gen[:, :, t], gt[:, :, t]))
    return [float(np.mean(d)) if d else float('nan') for d in by_frame]


def main():
    results = {}
    for tag, d in EVAL_DIRS.items():
        if not os.path.isdir(d):
            print(f'[skip] {tag}: dir not found -> {d}'); continue
        curve = per_frame_curve(d)
        results[tag] = curve
        print(f'{tag}: ' + '  '.join(f'f{t}={v:.3f}' for t, v in enumerate(curve)))

    if not results:
        print('No eval dirs available.'); return

    plt.figure(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(results)))
    for (label, vals), color in zip(results.items(), colors):
        plt.plot(range(T_FRAMES), vals, marker='o', linewidth=2,
                 markersize=7, label=label, color=color)
    plt.xlabel('Frame index (0 = i2v-conditioned, 1-12 = generated)')
    plt.ylabel('Per-frame 1 - SSIM (lower = better)')
    plt.title('Per-frame structural quality across training protocols')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f'Saved {OUT_PNG}')


if __name__ == '__main__':
    main()