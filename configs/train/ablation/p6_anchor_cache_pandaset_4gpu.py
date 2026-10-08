# Ablation P6: P5 (frame-aligned REG) + Anchor KV Cache on PandaSet — 4-GPU rerun
#
# WHY RERUN: the original 2-GPU P6 run (260906_170609) trained with a DEAD anchor
# cache — inject() looked up cache[layer_idx] but the anchor was stored under
# cache[anchor_layer], so injection never fired and all gates stayed at init
# (-6.0, sigmoid~0.0025) for 3000+ steps. anchor_cache.py is now fixed; this run
# trains the mechanism for real.
#
# Single ablation variable vs P5: use_anchor_cache=True. Everything else matches
# P5's protocol exactly (verified against p5_reg_frame_aligned_pandaset.py):
#   - AdamW (not LeanAdafactor — 4 GPUs hold the 11.2GB/rank fp32 fetch peak)
#   - num_frames=1, 224x400 (do NOT raise frames: P5 trained 1 frame; changing it
#     would confound the ablation)
#   - accumulation_steps=1 on 4 GPUs == P5's 2 GPUs x accumulation 2 (effective
#     batch 4 in both cases)
#   - init from baseline, seed 42, lr 1e-4, ckpt_every 500
_base_ = [
    "../../base/mmdit_config.py",
    "../../base/vae_config.py",
    "../../base/data_config.py",
]

use_multi_level_noise = True
traj_ctrl = True
use_depth = True
use_seg = True
is_multi_view = True
use_reg = True

dinov3_repo_dir = "dinov3"
dinov3_ckpt_path = "dinov3/dinov3_vitb16_pretrain.pth"
dinov3_on_cpu = False  # 4-GPU headroom (~8GB/rank freed vs 2-GPU) — DINOv3 on GPU

# Frame-aligned REG (identical to P5)
reg_use_depth_cls = True
reg_use_seg_cls = True
reg_use_align_loss = True
reg_frame_aligned = True

# Dataset
num_frames = 1
bucket_config = {
    "224x400r": {
        1: (1.0, 1),
    },
}

dataset = dict(
    type="nuscenes_video",
    pkl_path="/data1/qjyuan/OmniNWM/data/pandaset_pkl/pandaset_interp_12Hz_infos_train.pkl",
    dataset_name="pandaset",
    depth_seg_root="/data1/qjyuan/OmniNWM/data/pandaset_depth_seg",
    use_depth=use_depth,
    use_seg=use_seg,
    traj_ctrl=traj_ctrl,
    num_frames=num_frames,
    video_attr_list=[
        dict(height=224, width=400, frames=1),
    ],
)

condition_config = dict(
    i2v_head=5,
)

in_channels = 192
grad_ckpt_settings = (100, 100)
model = dict(
    from_pretrained=None,
    grad_ckpt_settings=grad_ckpt_settings,
    in_channels=in_channels,
    use_depth=use_depth,
    use_seg=use_seg,
    use_multi_level_noise=use_multi_level_noise,
    use_reg=use_reg,
    # ===== Ablation variable: Anchor KV Cache (NOW ACTIVE after bug fix) =====
    use_anchor_cache=True,
    anchor_cache_layer=0,                    # extract anchor from double block 0
    anchor_inject_double_layers=[4, 8, 12, 16],
    anchor_inject_single_layers=[5, 15, 25, 35],
    anchor_downsample=8,                     # 224x400 → 350 tokens/frame → ~44 anchors
    anchor_tokens=128,
    # ==========================================================================
    reg_use_align_loss=reg_use_align_loss,
    reg_frame_aligned=reg_frame_aligned,
)

is_causal_vae = True
lr = 1e-4
eps = 1e-15

# AdamW — matches P5 exactly (the 2-GPU run's LeanAdafactor was a hardware
# workaround that added an optimizer confound). 4 GPUs: fp32 state fetch peak
# 11.2GB/rank; 22.5 (model) + 5.6 (grad shard) + 11.2 (fetch) = ~39.3GB < 47GB.
optim = dict(
    cls="AdamW",
    lr=lr,
    eps=eps,
    weight_decay=0.0,
    # Gate dynamics: 8 scalar gates init at -6.0 (sigmoid~0.0025). At lr=1e-4
    # they can only traverse 0.5 logit in 5000 steps (sigmoid stays ~0.004) —
    # the cache could never activate. 100x lr (1e-2) lets a gate fully open in
    # ~600 steps IF gradients push it that way (AdamW step ≈ lr for a scalar).
    anchor_gate_lr_mult=100,
)

warmup_steps = 0
update_warmup_steps = True
grad_clip = 1.0
accumulation_steps = 1  # 4 GPUs x 1 == P5's 2 GPUs x 2 (effective batch 4)
ema_decay = None

# CPU budget: 32GB/GPU policy. AdamW states offloaded to CPU = 22GB/rank.
# Keep workers modest to avoid the known oom_kill trap from the 6-GPU runs.
prefetch_factor = 2
num_workers = 1
num_bucket_build_workers = 1

dtype = "bf16"
plugin = "zero2"
plugin_config = dict(
    reduce_bucket_size_in_m=64,  # P5's setting; 4-GPU headroom allows it
    overlap_allgather=False,
    cpu_offload=True,
    master_weights=False,
)

grad_checkpoint = True
pin_memory_cache_pre_alloc_numels = None
async_io = False

seed = 42
outputs = "./outputs/ablation_pandaset/p6_anchor_cache_4gpu"
epochs = 5000
log_every = 1
ckpt_every = 500
keep_n_latest = 200000
wandb_project = "omninwm_ablation_pandaset"
save_master_weights = True
load_master_weights = True
load = "/data1/qjyuan/OmniNWM/outputs/ablation_pandaset/p6_anchor_cache_4gpu/260908_145349-train_ablation_p6_anchor_cache_pandaset_4gpu/epoch0-global_step500"
# Resume run 260908_145349 (gate fixes active: fp32 storage + colossalai patches).
# That run started before anchor_gate_lr_mult existed, so its first 500 steps
# trained gates at lr=1e-4 (nearly frozen); from here on gates get lr=1e-2.
# start_from_scratch=False restores model + sampler state (data order preserved);
# optimizer moments restart from zero (checkpoints are saved without optimizer
# state — see omninwm/utils/ckpt.py load()).
start_from_scratch = False
