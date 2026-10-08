"""Per-group per-view FVD at i2v frame (f0):
- 取每个clip的f0（6 view各1张图）作为样本
- 192 clip / 16 一组 = 12组；每view独立算FVD
- 6 view取平均 → 每个group一个FVD
- 画 per-group FVD 曲线（P6 vs Baseline）

数据：
- Baseline: 36 clip → 2 组真实数据
- P6 step20k: 200 clip → 12 组
"""

import glob
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import linalg
from torchvision.models import inception_v3
from torchvision import transforms

EVAL_DIRS = {
    'P6 step20k': '/data1/qjyuan/OmniNWM/outputs/ablation_pandaset/p6_anchor_cache_4gpu_eval_step20000/video/eval_tensors',
    'Baseline':   '/data1/qjyuan/OmniNWM/outputs/infer_baseline/video/eval_tensors',
}

OUT_PNG = 'tools/fig_per_group_fvd.png'
GROUP_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Inception(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = inception_v3(weights='DEFAULT', aux_logits=True)
        m.eval(); m.to(DEVICE)
        self.m = m
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

    @torch.no_grad()
    def feats(self, frame):
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        if frame.shape[-1] != 299:
            frame = torch.nn.functional.interpolate(
                frame, size=(299, 299), mode='bilinear', align_corners=False
            )
        x = self.norm(frame).to(DEVICE)
        x = self.m.Conv2d_1a_3x3(x); x = self.m.Conv2d_2a_3x3(x); x = self.m.Conv2d_2b_3x3(x)
        x = torch.nn.functional.max_pool2d(x, 3, stride=2)
        x = self.m.Conv2d_3b_1x1(x); x = self.m.Conv2d_4a_3x3(x)
        x = torch.nn.functional.max_pool2d(x, 3, stride=2)
        x = self.m.Mixed_5b(x); x = self.m.Mixed_5c(x); x = self.m.Mixed_5d(x)
        x = self.m.Mixed_6a(x); x = self.m.Mixed_6b(x); x = self.m.Mixed_6c(x); x = self.m.Mixed_6d(x)
        x = self.m.Mixed_6e(x); x = self.m.Mixed_7a(x); x = self.m.Mixed_7b(x); x = self.m.Mixed_7c(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        return x.view(-1).cpu()


def frechet(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def view0_features(eval_dir, inception, max_clips=None):
    """返回 [N, V, 2048]"""
    g_files = sorted(glob.glob(os.path.join(eval_dir, '[0-9]*_rgb.pt')))
    g_files = [f for f in g_files if '_gt_' not in f]
    if max_clips:
        g_files = g_files[:max_clips]
    gen_feats, gt_feats = [], []
    for gf in tqdm(g_files, desc=f'f0 {os.path.basename(eval_dir)}', leave=False):
        gtf = gf.replace('_rgb.pt', '_gt_rgb.pt')
        if not os.path.exists(gtf):
            continue
        gen = torch.load(gf, map_location='cpu', weights_only=True).float()
        gt = torch.load(gtf, map_location='cpu', weights_only=True).float()
        V = gen.shape[0]
        gen_v, gt_v = [], []
        for v in range(V):
            gen_v.append(inception.feats(gen[v, :, 0]))
            gt_v.append(inception.feats(gt[v, :, 0]))
        gen_feats.append(torch.stack(gen_v))
        gt_feats.append(torch.stack(gt_v))
    return torch.stack(gen_feats), torch.stack(gt_feats)


def per_view_per_group_fvd(gen_feats, gt_feats, group_size=16):
    """对每个view、每group独立算FVD → 返回 list[(view, g_idx, fvd)]"""
    N, V = gen_feats.shape[:2]
    n_groups = N // group_size
    out = []
    for v in range(V):
        for g in range(n_groups):
            s = g * group_size
            e = s + group_size
            gv = gen_feats[s:e, v, :].numpy()
            tv = gt_feats[s:e, v, :].numpy()
            mg, sg = gv.mean(0), np.cov(gv, rowvar=False)
            mt, st = tv.mean(0), np.cov(tv, rowvar=False)
            out.append((v, g, frechet(mg, sg, mt, st)))
    return out


def main():
    print(f'Device: {DEVICE}, group_size={GROUP_SIZE}')
    inception = Inception()
    results = {}
    for tag, d in EVAL_DIRS.items():
        if not os.path.isdir(d):
            print(f'[skip] {tag}'); continue
        g, t = view0_features(d, inception)
        per_vg = per_view_per_group_fvd(g, t, GROUP_SIZE)
        # 每group: 6 view的FVD平均
        groups = {}
        for v, gi, fvd in per_vg:
            groups.setdefault(gi, []).append(fvd)
        group_curve = [float(np.mean(groups[gi])) for gi in sorted(groups)]
        results[tag] = group_curve
        print(f'{tag}: {g.shape[0]} clips → {len(group_curve)} groups')
        print(f'  per-group FVD: ' + '  '.join(f'g{gi}={v:.1f}' for gi, v in enumerate(group_curve)))

    if not results: return

    plt.figure(figsize=(9, 5))
    colors = {'P6 step20k': '#d62728', 'Baseline': '#1f77b4'}
    for tag, curve in results.items():
        x = list(range(len(curve)))
        plt.plot(x, curve, marker='o', linewidth=2, markersize=8,
                 label=f'{tag} (mean={np.mean(curve):.2f})',
                 color=colors.get(tag, 'gray'))
    plt.xlabel(f'Group index (each = {GROUP_SIZE} clips of f0, 6-view averaged)')
    plt.ylabel('FVD (per view × per group, 6-view mean)')
    plt.title('Per-group FVD at i2v-conditioned frame (f0)\n'
              'Each group = 16 consecutive clips\'s f0, 6 views averaged')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f'Saved {OUT_PNG}')


if __name__ == '__main__':
    main()