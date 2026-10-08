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
# DINOv3 enabled: REG align loss is the core objective of stage3.
reg_use_depth_cls = True
reg_use_seg_cls = True
dinov3_on_cpu = True  # 2-GPU run ~90MB short at optimizer step; DINOv3 on CPU fixes it

dinov3_repo_dir = "dinov3"
dinov3_ckpt_path = "dinov3/dinov3_vitb16_pretrain.pth"

# View order (from data_config.py, must be explicit since we don't inherit)
view_order = ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT","CAM_BACK","CAM_BACK_LEFT"]

# Dataset settings - 1 frame at 112x200 for 2-GPU memory envelope.
# 2-GPU floor: model 22.5GB + grad shard 11.2GB + Adafactor state ~0.3GB = ~34GB
# + DINOv3 ~1GB + VAE activations → ~38GB < 47GB. (AdamW floor was 45GB → OOM.)
num_frames = 1
bucket_config = {
    "112x200r": {
        1:  (1.0, 1),
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
        dict(height=112, width=200, frames=1),
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
    use_anchor_cache = True,
    anchor_cache_layer = 0,                   # Extract anchor from double block 0
    anchor_inject_double_layers = [8, 16],    # Inject at double blocks 8 and 16
    anchor_inject_single_layers = [15, 35],   # Inject at single blocks 15 and 35
    anchor_downsample = 16,                   # Spatial stride for anchor tokens (reduces cross-attn cost)
    anchor_tokens = 128,                      # Max number of anchor tokens
)

is_causal_vae = True
# Optimization settings
# Adafactor: factored 2nd moment → optimizer state ~0.3GB (AdamW needs 11.2GB shard
# on 2 GPU, which OOMs at 46.75/47.38GB). Updates are Adam-like adaptive (≈lr),
# 1e5x larger than SGD's lr*grad → not killed by bf16 rounding.
# lr=1e-4 (stage2-proven): keeps per-step update near bf16 ulp of typical weights.
lr = 1e-4

optim = dict(
    cls="LeanAdafactor",
    lr=lr,
    weight_decay=0.0,
)

warmup_steps = 500  # Warmup for stage3 fine-tuning
update_warmup_steps = True

grad_clip = 1.0
accumulation_steps = 4  # effective batch = 4 * num_gpus
ema_decay = None

# Acceleration settings
prefetch_factor = None  # default when num_workers=0
num_workers = 0
num_bucket_build_workers = 1

dtype = "bf16"
plugin = "zero2"
plugin_config = dict(
    reduce_bucket_size_in_m=8,  # small bucket for 2 GPU
    overlap_allgather=False,
    cpu_offload=True,
    master_weights=False,
)

grad_checkpoint = True
pin_memory_cache_pre_alloc_numels = None
async_io = False

# Other settings
seed = 42
outputs = "./outputs/stage3_anchor_kv_2gpu"  # separate dir: 4-GPU idle job resumes from stage3_anchor_kv
epochs = 10000
log_every = 1
ckpt_every = 200
keep_n_latest = 200000
wandb_project = "omninwm"
save_master_weights = True
load_master_weights = True
load = "/data1/qjyuan/OmniNWM/outputs/stage3_anchor_kv_2gpu/260831_174549-train_stage_3_2gpu/epoch0-global_step6200"
start_from_scratch = True  # ckpts save model only (no optimizer state); LeanAdafactor re-warms its EMA in ~100 steps
