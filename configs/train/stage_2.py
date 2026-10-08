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
# num_frames = 1
# bucket_config = {
#     "224x400r": {
#         1:  (1.0, 1),
#     },
# }

# dataset = dict(
#     use_depth=use_depth,
#     use_seg=use_seg,
#     traj_ctrl=traj_ctrl,
#     num_frames = num_frames,
#     video_attr_list = [
#         dict(height=224, width=400, frames= 1),
#     ],
#     max_samples = 500,
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
#     reg_dim = 768,             # DINOv3 ViT-B 的特征维度
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

# # === 新增 REG Loss 权重 ===
# lambda_align = 0.1  # 对应 REG 论文中的平衡系数，建议初始设为 0.1

# warmup_steps = 0
# update_warmup_steps = True

# grad_clip = 1.0
# accumulation_steps = 1
# ema_decay = None

# # Acceleration settings
# prefetch_factor = 8
# num_workers = 4
# num_bucket_build_workers = 4

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
# outputs = "./outputs/stage2"
# epochs = 10000
# log_every = 1
# ckpt_every = 100
# keep_n_latest = 200000
# wandb_project = "omninwm"
# save_master_weights = True
# load_master_weights = True
# load = "/data1/qjyuan/OmniNWM/outputs/stage12/260331_104248-train_stage_1/epoch13-global_step6700"
# start_from_scratch = True

_base_ = [
    "../base/mmdit_config.py",
    "../base/vae_config.py",
    "../base/data_config.py",
]

use_multi_level_noise = True

traj_ctrl = True
use_depth = True
use_seg = True
is_multi_view = True
use_reg = True

dinov3_repo_dir = "dinov3"
dinov3_ckpt_path = "dinov3/dinov3_vitb16_pretrain.pth"

# Dataset settings
num_frames = 5
bucket_config = {
    "224x400r": {
        5:  (1.0, 1),
    },
}

dataset = dict(
    pkl_path = "/data1/qjyuan/OmniNWM/data/nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl",
    use_depth=use_depth,
    use_seg=use_seg,
    traj_ctrl=traj_ctrl,
    num_frames = num_frames,
    video_attr_list = [
        dict(height=224, width=400, frames=5),
    ],
    max_samples = 500,
)

condition_config = dict(
    i2v_head=5,  # train i2v (image as first frame) with weight 5
)

# Define model components
# REG模式下depth/seg latent仍然拼接到x_0中，所以in_channels需要包含它们的通道数
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
    # Anchor KV Cache 配置
    use_anchor_cache = True,
    anchor_cache_layer = 0,  # 在第0个 Double Block 提取第一帧 K/V
    anchor_inject_double_layers = [8, 16],  # 减少注入层以节省显存
    anchor_inject_single_layers = [15, 35],  # 减少注入层以节省显存
    anchor_downsample = 16,  # 更激进的下采样，anchor tokens更少
    anchor_tokens = 128,
)

is_causal_vae = True
# Optimization settings
lr = 1e-4
eps = 1e-15

optim = dict(
    cls="AdamW",
    lr=lr,
    eps=eps,
    weight_decay=0.0,
)

warmup_steps = 0
update_warmup_steps = True

grad_clip = 1.0
accumulation_steps = 2
ema_decay = None

# Acceleration settings
prefetch_factor = 8
num_workers = 4
num_bucket_build_workers = 4

dtype = "bf16"
plugin = "zero2"
plugin_config = dict(
    reduce_bucket_size_in_m=64,
    overlap_allgather=False,
    cpu_offload=True,  # 将优化器状态offload到CPU，节省GPU显存
)

grad_checkpoint = True
pin_memory_cache_pre_alloc_numels = None
async_io = False

# Other settings
seed = 42
outputs = "./outputs/stage2"
epochs = 10000
log_every = 1
ckpt_every = 100
keep_n_latest = 200000
wandb_project = "omninwm"
save_master_weights = True
load_master_weights = True
load = "/data1/qjyuan/OmniNWM/pretrained/baseline/epoch0-global_step10600"
start_from_scratch = True