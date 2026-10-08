# _base_ = [
#     "../base/mmdit_config.py",
#     "../base/vae_config.py",
#     "../base/data_config.py",
# ]

# use_multi_level_noise = True

# traj_ctrl = True
# use_depth = True
# use_seg = True
# is_multi_view = True

# # Dataset settings
# num_frames = 33
# bucket_config = {
#     "224x400r": {
#         33:  (1.0, 1),
#     },
#     "448x800r": {
#         17:  (1.0, 1),
#     },
# }

# dataset = dict(
#     use_depth=use_depth,
#     use_seg=use_seg,
#     traj_ctrl=traj_ctrl,
#     num_frames = num_frames,
#     video_attr_list = [
#         dict(height=224, width=400, frames= 33),
#         dict(height=448, width=800, frames= 17),
#     ]
# )

# condition_config = dict(
#     i2v_head=5,  # train i2v (image as first frame) with weight 5
# )

# # Define model components
# if use_depth and use_seg:
#     in_channels = 192
# elif not use_depth and use_seg:
#     in_channels = 128
# elif use_depth and not use_seg:
#     in_channels = 128
# else:
#     in_channels = 64
# grad_ckpt_settings = (100, 100)  # set the grad checkpoint settings
# model = dict(
#     from_pretrained = None,
#     grad_ckpt_settings=grad_ckpt_settings,
#     in_channels = in_channels,
#     use_depth = use_depth,
#     use_seg = use_seg,
#     use_multi_level_noise = use_multi_level_noise,
# )

# is_causal_vae = True
# # Optimization settings
# lr = 1e-4
# eps = 1e-15

# optim = dict(
#     cls="HybridAdam",
#     lr=lr,
#     eps=eps,
#     weight_decay=0.0,
#     adamw_mode=True,
# )

# warmup_steps = 0
# update_warmup_steps = True

# grad_clip = 1.0
# accumulation_steps = 1
# ema_decay = None

# # Acceleration settings
# prefetch_factor = 2
# num_workers = 16
# num_bucket_build_workers = 16

# dtype = "bf16"
# plugin = "zero2"
# plugin_config = dict(
#     reduce_bucket_size_in_m=128,
#     overlap_allgather=False,
# )

# grad_checkpoint = True
# pin_memory_cache_pre_alloc_numels = None
# async_io = False

# # Other settings
# seed = 42
# outputs = "./outputs/stage3"
# epochs = 10000
# log_every = 1
# ckpt_every = 100
# keep_n_latest = 200000
# wandb_project = "omninwm"
# save_master_weights = True
# load_master_weights = True
# load = load = "/path/to/stage2"
# start_from_scratch = True

_base_ = [
    "../base/mmdit_config.py",
    "../base/vae_config.py",
    # NOT inheriting data_config.py to avoid bucket_config merge with 33-frame buckets
]

use_multi_level_noise = True

traj_ctrl = True
use_depth = True
use_seg = True
is_multi_view = True
use_reg = True
# DINOv3 restored: REG align loss is the core objective of stage3.
# (2-card run had it disabled + SGD → align=0 and updates below bf16 rounding.)
reg_use_depth_cls = True
reg_use_seg_cls = True

dinov3_repo_dir = "dinov3"
dinov3_ckpt_path = "dinov3/dinov3_vitb16_pretrain.pth"

# View order (from data_config.py, must be explicit since we don't inherit)
view_order = ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT","CAM_BACK","CAM_BACK_LEFT"]

# Dataset settings - 4 frames at 112x200, proven 4-GPU config (Job 93).
num_frames = 4
bucket_config = {
    "112x200r": {
        4:  (1.0, 1),
    },
}

dataset = dict(
    type="nuscenes_video",
    pkl_path = "/data1/qjyuan/OmniNWM/data/nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl",
    transform_name="resize_crop",
    fps_max=24,
    vmaf=False,
    memory_efficient=False,
    view_order=view_order,
    use_depth=use_depth,
    use_seg=use_seg,
    traj_ctrl=traj_ctrl,
    num_frames = num_frames,
    max_depth = 100,
    video_attr_list = [
        dict(height=112, width=200, frames=4),
    ]
)

condition_config = dict(
    i2v_head=5,  # train i2v (image as first frame) with weight 5
)

# Define model components
# Stage3 must use in_channels=192 to match stage2 checkpoint (depth+seg latent concatenated)
if use_depth and use_seg:
    in_channels = 192
elif not use_depth and use_seg:
    in_channels = 128
elif use_depth and not use_seg:
    in_channels = 128
else:
    in_channels = 64
grad_ckpt_settings = (100, 100)
model = dict(
    from_pretrained = None,
    grad_ckpt_settings=grad_ckpt_settings,
    in_channels = in_channels,
    use_depth = use_depth,
    use_seg = use_seg,
    use_multi_level_noise = use_multi_level_noise,
    use_reg = use_reg,
    # Anchor KV Cache for spatiotemporal consistency
    # Inspired by INSPATIO-WORLD's Spatiotemporal Cache mechanism:
    #   - Extract first frame's intermediate K/V as spatial anchor (at layer 0)
    #   - Inject anchor into subsequent frames via gated cross-attention
    #   - Provides persistent "spatial memory" for temporal consistency
    # Differences from INSPATIO-WORLD:
    #   - INSPATIO-WORLD: autoregressive chunk-to-chunk cache for chunked generation
    #   - OmniNWM: frame-anchored cache for full-sequence generation (first frame as anchor)
    #   - Gate initialization: sigmoid(-6.0) ≈ 0.0025 → cache starts nearly disabled
    #   - Gradually learned during training → avoids disrupting pretrained features
    use_anchor_cache = True,
    anchor_cache_layer = 0,                   # Extract anchor from double block 0
    anchor_inject_double_layers = [8, 16],    # Inject at double blocks 8 and 16
    anchor_inject_single_layers = [15, 35],   # Inject at single blocks 15 and 35
    anchor_downsample = 16,                   # Spatial stride for anchor tokens (reduces cross-attn cost)
    anchor_tokens = 128,                      # Max number of anchor tokens
)

is_causal_vae = True
# Optimization settings
lr = 5e-5  # Lower than stage2 (1e-4) for fine-tuning stability
eps = 1e-15

# AdamW restored. SGD on 2-GPU was a dead end: update = lr*grad ≈ 2.4e-10 per param,
# ~1e5x smaller than bf16 rounding (~4e-5) → weights never changed, loss flat for 23k steps.
optim = dict(
    cls="AdamW",
    lr=lr,
    eps=eps,
    weight_decay=0.0,
)

warmup_steps = 500  # Warmup for stage3 fine-tuning
update_warmup_steps = True

grad_clip = 1.0
accumulation_steps = 4  # effective batch = 4 * num_gpus, reduces peak memory per step
ema_decay = None

# Acceleration settings
prefetch_factor = None  # default when num_workers=0 (no extra pinned memory)
num_workers = 0  # No dataloader workers: avoid spawning 6 extra Python procs (each ~6GB)
num_bucket_build_workers = 1

dtype = "bf16"
plugin = "zero2"  # ZeRO2 + cpu_offload: optimizer on CPU to free GPU for VAE
plugin_config = dict(
    reduce_bucket_size_in_m=32,
    overlap_allgather=False,
    cpu_offload=True,
    master_weights=False,
)

grad_checkpoint = True
pin_memory_cache_pre_alloc_numels = None
async_io = False

# Other settings
seed = 42
outputs = "./outputs/stage3_anchor_kv"
epochs = 10000
log_every = 1
ckpt_every = 200  # Save every 200 steps: idle partition may be preempted anytime
keep_n_latest = 200000
wandb_project = "omninwm"
save_master_weights = True
load_master_weights = True
load = "/data1/qjyuan/OmniNWM/outputs/stage3_anchor_kv/260824_111141-train_stage_3/epoch0-global_step500"
start_from_scratch = True  # Checkpoint has no optimizer subdir; load model weights only