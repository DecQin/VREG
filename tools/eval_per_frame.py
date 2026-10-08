"""Per-frame quality metrics from saved inference eval_tensors.

Computes how generation quality degrades with frame index (the "later frames
get worse" curve) from the paired tensors saved by inference runs:
    eval_tensors/{idx}_rgb.pt     generated  [V, 3, T, H, W] in [0, 1]
    eval_tensors/{idx}_gt_rgb.pt  ground truth, same shape

For each frame position t (0..T-1) it pools PSNR / SSIM / LPIPS over all
samples and views, and reports a 95% bootstrap CI (resampled at the sample
level, since the V views of one sample are correlated).

Frame 0 is the i2v condition (GT first frame fed to the model), so all
models should overlap there; the interesting signal is the gap at t > 0.

Usage:
    python tools/eval_per_frame.py --eval_dir <...>/eval_tensors \
        --metrics psnr ssim lpips --device cuda --output per_frame.csv
    (lpips needs a GPU; psnr is fast anywhere; ssim is OK on CPU)
"""

import argparse
import glob
import os
import random

import numpy as np
import torch


def load_pairs(eval_dir):
    gen_files = sorted(glob.glob(os.path.join(eval_dir, "*_rgb.pt")))
    pairs = []
    for gf in gen_files:
        base = gf[: -len("_rgb.pt")]
        gtf = base + "_gt_rgb.pt"
        if os.path.exists(gtf):
            pairs.append((gf, gtf))
    return pairs


def psnr_batch(gen, gt):
    """gen, gt: [V, 3, H, W] in [0,1] -> psnr per view [V]"""
    mse = ((gen - gt) ** 2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True, help="eval_tensors directory")
    parser.add_argument("--metrics", nargs="+", default=["psnr"], choices=["psnr", "ssim", "lpips"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="output csv path")
    parser.add_argument("--max-samples", type=int, default=None, help="subsample for quick validation")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = load_pairs(args.eval_dir)
    if not pairs:
        raise FileNotFoundError(f"no *_rgb.pt / *_gt_rgb.pt pairs under {args.eval_dir}")
    if args.max_samples:
        pairs = pairs[: args.max_samples]
    n_samples = len(pairs)
    print(f"[per-frame eval] {n_samples} paired samples from {args.eval_dir}")

    lpips_fn = None
    if "lpips" in args.metrics:
        import lpips

        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(args.device)
        lpips_fn.eval()

    if "ssim" in args.metrics:
        from skimage.metrics import structural_similarity

    # first pass to get T
    gen0 = torch.load(pairs[0][0], map_location="cpu")
    T = gen0.shape[2]
    V = gen0.shape[0]
    print(f"[per-frame eval] views={V}, frames={T}")

    # per-metric: array [n_samples, V, T]
    acc = {m: np.zeros((n_samples, V, T), dtype=np.float64) for m in args.metrics}

    for i, (gf, gtf) in enumerate(pairs):
        gen = torch.load(gf, map_location="cpu").float().clamp(0, 1)   # [V,3,T,H,W]
        gt = torch.load(gtf, map_location="cpu").float().clamp(0, 1)
        for t in range(T):
            g_t = gen[:, :, t]  # [V,3,H,W]
            gt_t = gt[:, :, t]
            if "psnr" in acc:
                acc["psnr"][i, :, t] = psnr_batch(g_t, gt_t).numpy()
            if "ssim" in acc:
                g_np = g_t.permute(0, 2, 3, 1).numpy()
                gt_np = gt_t.permute(0, 2, 3, 1).numpy()
                for v in range(V):
                    acc["ssim"][i, v, t] = structural_similarity(
                        g_np[v], gt_np[v], channel_axis=2, data_range=1.0
                    )
            if "lpips" in acc:
                with torch.no_grad():
                    d = lpips_fn(
                        g_t.to(args.device), gt_t.to(args.device)
                    ).flatten().cpu().numpy()  # [V]
                acc["lpips"][i, :, t] = d
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_samples} samples done")

    # aggregate per frame with bootstrap CI over samples
    rng = np.random.default_rng(args.seed)
    run_name = os.path.basename(os.path.dirname(os.path.dirname(args.eval_dir.rstrip("/"))))

    lines = ["run,frame_idx,metric,mean,ci_lo,ci_hi,n_samples"]
    for m in args.metrics:
        for t in range(T):
            vals = acc[m][:, :, t].reshape(n_samples, V)
            per_sample = vals.mean(axis=1)  # mean over views -> [n_samples]
            mean = per_sample.mean()
            boots = np.empty(args.bootstrap)
            for b in range(args.bootstrap):
                idx = rng.integers(0, n_samples, n_samples)
                boots[b] = per_sample[idx].mean()
            lo, hi = np.percentile(boots, [2.5, 97.5])
            lines.append(f"{run_name},{t},{m},{mean:.5f},{lo:.5f},{hi:.5f},{n_samples}")

    out = args.output or os.path.join(os.path.dirname(args.eval_dir), "per_frame_metrics.csv")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[per-frame eval] wrote {out}")

    # quick console summary
    for m in args.metrics:
        means = [float(l.split(",")[3]) for l in lines[1:] if f",{m}," in l]
        print(f"  {m}: frame0={means[0]:.4f}  frame{len(means)-1}={means[-1]:.4f}  "
              f"(degradation {means[0] - means[-1]:+.4f})")


if __name__ == "__main__":
    main()
