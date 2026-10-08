"""Per-clip view0-FVD: 每个clip取 f0（6 view 各 1 帧）作为
"6-frame mini-video"算 FVD——衡量 i2v 条件帧的多视角一致性。

P6 step20k vs Baseline 对比图。
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

OUT_PNG = 'tools/fig_view0_fvd.png'
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


def view0_features(eval_dir, inception):
    """返回 [N, V, 2048] ——N是clip数，V=6是view"""
    g_files = sorted(glob.glob(os.path.join(eval_dir, '[0-9]*_rgb.pt')))
    g_files = [f for f in g_files if '_gt_' not in f]
    gen_feats, gt_feats = [], []
    for gf in tqdm(g_files, desc=f'view0 {os.path.basename(eval_dir)}', leave=False):
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


def per_view_fvd(gen_feats, gt_feats):
    """每个 view 独立算 Fréchet 距离 → 6 个值 → 取平均"""
    V = gen_feats.shape[1]
    vals = []
    for v in range(V):
        g = gen_feats[:, v, :].numpy()  # [N, 2048]
        t = gt_feats[:, v, :].numpy()
        mg, sg = g.mean(0), np.cov(g, rowvar=False)
        mt, st = t.mean(0), np.cov(t, rowvar=False)
        vals.append(frechet(mg, sg, mt, st))
    return vals  # 6 个 view 各自的 FVD


def main():
    print(f'Device: {DEVICE}')
    inception = Inception()
    results = {}
    for tag, d in EVAL_DIRS.items():
        if not os.path.isdir(d):
            print(f'[skip] {tag}: dir not found'); continue
        g, t = view0_features(d, inception)
        fvds = per_view_fvd(g, t)  # 6 个值
        mean_v = float(np.mean(fvds))
        results[tag] = {'per_view': fvds, 'mean': mean_v}
        per_view_str = '  '.join(f'v{v}={f:.2f}' for v, f in enumerate(fvds))
        print(f'{tag}: {g.shape[0]} clips, mean FVD = {mean_v:.2f}  | {per_view_str}')

    if not results: return

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    n_views = 6
    x = np.arange(n_views)
    width = 0.8 / len(names)
    colors = ['#d62728', '#1f77b4', '#2ca02c'][:len(names)]
    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * width
        bars = ax.bar(x + offset, results[name]['per_view'], width,
                      label=f"{name} (mean={results[name]['mean']:.2f})",
                      color=colors[i], alpha=0.85, edgecolor='black')
        for b, v in zip(bars, results[name]['per_view']):
            ax.text(b.get_x() + b.get_width()/2, v + 1, f'{v:.1f}',
                     ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'view {v}' for v in range(n_views)])
    ax.set_ylabel('FVD (per view, computed independently)')
    ax.set_title('Per-view FVD at i2v-conditioned frame (f0) — 6 views averaged')
    ax.legend(loc='best', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f'Saved {OUT_PNG}')


if __name__ == '__main__':
    main()