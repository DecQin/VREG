# Ablation P7b: Multi-frame control (13 frames, NO cache) on PandaSet — 4 GPU
#
# CONTROL RUN for P7 (p7_anchor_cache_multiframe_pandaset_4gpu.py).
# Identical multi-frame protocol (13 frames @ 224x400, i2v_head condition,
# same init/optimizer/seed/budget) with use_anchor_cache=False. Together:
#   - P7 vs P6: does the multi-frame protocol + active cache fix FVD?
#   - P7 vs P7b: what does the anchor cache contribute on top of multi-frame
#     training alone? (attribution for the ablation table)
# Background: P0-P6 all fine-tuned at num_frames=1, which (1) made the anchor
# cache degenerate (gates learned closed: sigmoid 0.0002-0.0029 at step 10k),
# (2) eroded the base model's temporal prior (the FVD killer), and (3) skipped
# the i2v_head condition branch entirely. See the P7 header for details.
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
dinov3_on_cpu = False  # DINOv3 fwd is no_grad target extraction; 13 frames OK on GPU.
                        # If CUDA OOM at T=13, set True first (proven fallback).

# Frame-aligned REG (identical to P5/P6)
reg_use_depth_cls = True
reg_use_seg_cls = True
reg_use_align_loss = True
reg_frame_aligned = True

# Dataset — 13 frames to match the 13-frame eval protocol exactly.
num_frames = 13
bucket_config = {
    "224x400r": {
        13: (1.0, 1),
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
        dict(height=224, width=400, frames=13),
    ],
)

# With T>1 the video branch of prepare_visual_condition_causal now fires:
# first-frame mask + GT latent every sample, matching inference exactly.
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
    # ===== Control: NO anchor cache (P7's only differing variable) =====
    use_anchor_cache=False,
    # ====================================================================
    reg_use_align_loss=reg_use_align_loss,
    reg_frame_aligned=reg_frame_aligned,
)

is_causal_vae = True
lr = 1e-4
eps = 1e-15

# AdamW as in P5/P6/P7. No anchor gates in this control, so
# anchor_gate_lr_mult is a no-op (kept for config parity with P7).
optim = dict(
    cls="AdamW",
    lr=lr,
    eps=eps,
    weight_decay=0.0,
    anchor_gate_lr_mult=100,
)

warmup_steps = 0
update_warmup_steps = True
grad_clip = 1.0
accumulation_steps = 1  # 4 GPUs x 1 sample x 13 frames (effective batch 4 clips)
ema_decay = None

# CPU budget: 32GB/GPU policy. 13-frame clips raise per-worker memory; keep
# workers minimal to avoid the known oom_kill trap.
prefetch_factor = 2
num_workers = 1
num_bucket_build_workers = 1

dtype = "bf16"
plugin = "zero2"
plugin_config = dict(
    reduce_bucket_size_in_m=32,  # smaller buckets than P6: T=13 headroom
    overlap_allgather=False,
    cpu_offload=True,
    master_weights=False,
)

grad_checkpoint = True
pin_memory_cache_pre_alloc_numels = None
async_io = False

seed = 42
outputs = "./outputs/ablation_pandaset/p7b_multiframe_nocache_4gpu"
epochs = 5000
log_every = 1
ckpt_every = 500
keep_n_latest = 200000
wandb_project = "omninwm_ablation_pandaset"
save_master_weights = True
load_master_weights = True
# Fresh start from the clean multi-frame baseline (same init as P0-P6).
load = "/data1/qjyuan/OmniNWM/pretrained/baseline/epoch0-global_step10600"
start_from_scratch = True
