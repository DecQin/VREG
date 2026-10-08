# Ablation P6: P5 (frame-aligned REG) + Anchor KV Cache on PandaSet
# Single ablation variable vs P5: use_anchor_cache=True.
#   - Extracts first-frame K/V at double block 0, injects into designated layers
#     via gated cross-attention (gates init -6.0 ≈ near-zero injection at start).
#   - ~8 scalar gate params only: capacity is negligible, gains come from mechanism.
# Anchor hyperparams MUST match the P6 inference config exactly (gates are keyed
# by layer index in ParameterDict — mismatched layer lists silently drop gates).
#
# NOTE (hardware-forced deviations from P5 protocol, same as stage3 2-GPU runs):
#   - optim: LeanAdafactor instead of AdamW — this node's 47GB GPUs cannot hold
#     AdamW's 11.2GB/rank optimizer fetch peak for the 11B model (P5 ran on an
#     80GB machine). lr unchanged (1e-4).
#   - num_workers=0 / master_weights=False / dinov3_on_cpu=True: 32GB-CPU-per-GPU
#     cluster policy (same envelope as stage_3_2gpu.py, which is proven here).
# For a strictly comparable ablation table, rerun P5 with these same settings
# (p5r) so the P5-vs-P6 pair differs ONLY in use_anchor_cache.
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
dinov3_on_cpu = True  # 2-GPU envelope (proven in stage_3_2gpu)

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
    # ===== Ablation variable: Anchor KV Cache =====
    use_anchor_cache=True,
    anchor_cache_layer=0,                    # extract anchor from double block 0
    anchor_inject_double_layers=[4, 8, 12, 16],
    anchor_inject_single_layers=[5, 15, 25, 35],
    anchor_downsample=8,                     # 224x400 → 350 tokens/frame → ~44 anchors
    anchor_tokens=128,
    # ==============================================
    reg_use_align_loss=reg_use_align_loss,
    reg_frame_aligned=reg_frame_aligned,
)

is_causal_vae = True
lr = 1e-4
eps = 1e-15

# LeanAdafactor (see header): factored 2nd moment → ~0.3GB optimizer state on 2 GPU.
# AdamW would need an 11.2GB/rank fetch peak → OOM on 47GB cards.
optim = dict(
    cls="LeanAdafactor",
    lr=lr,
    eps=eps,
    weight_decay=0.0,
)

warmup_steps = 0
update_warmup_steps = True
grad_clip = 1.0
accumulation_steps = 2
ema_decay = None

# 2-GPU CPU envelope (32GB/GPU policy): no workers, no master weights.
prefetch_factor = None
num_workers = 0
num_bucket_build_workers = 1

dtype = "bf16"
plugin = "zero2"
plugin_config = dict(
    reduce_bucket_size_in_m=8,
    overlap_allgather=False,
    cpu_offload=True,
    master_weights=False,
)

grad_checkpoint = True
pin_memory_cache_pre_alloc_numels = None
async_io = False

seed = 42
outputs = "./outputs/ablation_pandaset/p6_anchor_cache"
epochs = 5000
log_every = 1
ckpt_every = 500
keep_n_latest = 200000
wandb_project = "omninwm_ablation_pandaset"
save_master_weights = True
load_master_weights = True
load = "/data1/qjyuan/OmniNWM/pretrained/baseline/epoch0-global_step10600"
start_from_scratch = True  # baseline has no gate params; gates keep init (-6.0)
