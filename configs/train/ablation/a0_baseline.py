# Ablation A0: Baseline (no REG, no Anchor Cache)
# Pure depth/seg latent conditioning, no CLS injection, no alignment loss
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
use_reg = False  # No REG at all

# Dataset
num_frames = 1
bucket_config = {
    "224x400r": {
        1: (1.0, 1),
    },
}

dataset = dict(
    pkl_path="/data1/qjyuan/OmniNWM/data/nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl",
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
    use_anchor_cache=False,  # No Anchor Cache
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
accumulation_steps = 2
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

seed = 42
outputs = "./outputs/ablation/a0_baseline"
epochs = 5000
log_every = 1
ckpt_every = 500
keep_n_latest = 200000
wandb_project = "omninwm_ablation"
save_master_weights = True
load_master_weights = True
load = "/data1/qjyuan/OmniNWM/outputs/ablation/a0_baseline/260704_145435-train_ablation_a0_baseline/epoch0-global_step500"
start_from_scratch = True
