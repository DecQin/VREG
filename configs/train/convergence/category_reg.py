# Convergence Comparison: Category-aligned REG
# Global mean pool CLS → broadcast injection + global alignment loss
_base_ = [
    "../../base/mmdit_config.py",
    "../../base/vae_config.py",
    "../../base/data_config.py",
]

# == Variant identity ==
variant = "category_reg"

use_multi_level_noise = True
traj_ctrl = True
use_depth = True
use_seg = True
is_multi_view = True
use_reg = True

dinov3_repo_dir = "dinov3"
dinov3_ckpt_path = "dinov3/dinov3_vitb16_pretrain.pth"

# Category-aligned REG: global mean pool for both injection and alignment
reg_use_depth_cls = True
reg_use_seg_cls = True
reg_use_align_loss = True
reg_frame_aligned = False    # Global alignment loss
reg_frame_injection = False  # Global broadcast injection (NOT per-frame)

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
    use_anchor_cache=False,
    reg_use_align_loss=reg_use_align_loss,
    reg_frame_aligned=reg_frame_aligned,
    reg_frame_injection=reg_frame_injection,
)

is_causal_vae = True
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
accumulation_steps = 6  # 2 GPUs × 6 = 12 effective batch (matches 6-GPU × 2 setup)
ema_decay = None

prefetch_factor = 8
num_workers = 4
num_bucket_build_workers = 4

dtype = "bf16"
plugin = "zero2"
plugin_config = dict(
    reduce_bucket_size_in_m=64,
    overlap_allgather=False,
    cpu_offload=True,
)

grad_checkpoint = True
pin_memory_cache_pre_alloc_numels = None
async_io = False

# == Convergence-specific ==
seed = 42
outputs = "./outputs/convergence/category_reg/seed_42"
epochs = 5000
log_every = 1
ckpt_every = 500
keep_n_latest = 200000
save_master_weights = True
load_master_weights = True
start_from_scratch = True
load = "/data1/qjyuan/OmniNWM/pretrained/baseline/epoch0-global_step10600"

# == Convergence profiling ==
record_time = True
convergence_variant = "category_reg"
convergence_save_steps = [500, 1000, 2000, 5000, 10000, 20000, 25000, 30000]
