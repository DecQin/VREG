_base_ = [
    "../base/mmdit_config.py",
    "../base/vae_config.py",
    "../base/data_config.py",
]

use_multi_level_noise = True
traj_ctrl = True
use_depth = False
use_seg = False
is_multi_view = True

# Dataset settings
num_frames = 1
bucket_config = {
    "112x200r": {
        1:  (1.0, 1),
    },
    # "224x400r": {
    #     33:  (1.0, 1),
    # },
}

dataset = dict(
    use_depth=use_depth,
    use_seg=use_seg,
    traj_ctrl=traj_ctrl,
    num_frames = num_frames,
    video_attr_list = [
        dict(height=112, width=200, frames= 1),
        # dict(height=112, width=200, frames= 33),
        # dict(height=224, width=400, frames= 33),
    ],
    max_samples = 500,  # DEBUG: 限制样本数量，正式训练时可以移除或设置为 None
)

condition_config = dict(
    i2v_head=5,  # train i2v (image as first frame) with weight 5
)

# Define model components
if use_depth and use_seg:
    in_channels = 192
elif not use_depth and use_seg:
    in_channels = 128
elif use_depth and not use_seg:
    in_channels = 128
else:
    in_channels = 64
grad_ckpt_settings = (100, 100)  # set the grad checkpoint settings
model = dict(
    from_pretrained = 'pretrained/Open_Sora_v2.safetensors',
    grad_ckpt_settings=grad_ckpt_settings,
    in_channels = in_channels,
    use_depth = use_depth,
    use_seg = use_seg,
    use_multi_level_noise = use_multi_level_noise,
    reg_dim = 768,             # DINOv3 ViT-B 的特征维度
)

is_causal_vae = True
# Optimization settings
lr = 1e-4
eps = 1e-15

optim = dict(
    cls="HybridAdam",
    lr=lr,
    eps=eps,
    weight_decay=0.0,
    adamw_mode=True,
)

# === 新增 REG Loss 权重 ===
lambda_align = 0.1  # 对应 REG 论文中的平衡系数，建议初始设为 0.1

warmup_steps = 0
update_warmup_steps = True

grad_clip = 1.0
accumulation_steps = 1
ema_decay = None

# Acceleration settings
prefetch_factor = 8
num_workers = 4
num_bucket_build_workers = 4

dtype = "bf16"
plugin = "zero2" # 2
plugin_config = dict(
    reduce_bucket_size_in_m=128,
    overlap_allgather=False,
    cpu_offload=True,
)

grad_checkpoint = True
pin_memory_cache_pre_alloc_numels = None
async_io = False

# Other settings
seed = 42
outputs = "./outputs/stage12"
epochs = 10000
log_every = 1
ckpt_every = 100
keep_n_latest = 200000
wandb_project = "omninwm"
save_master_weights = True
load_master_weights = True

load = "./outputs/stage12/260330_141634-train_stage_1/epoch1-global_step700"