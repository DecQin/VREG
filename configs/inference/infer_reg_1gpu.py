calculate_reward = False
use_multi_level_noise = True
use_low_men_vae_infer = True
traj_ctrl = True
use_depth = True
use_seg = True
use_reg = True
dinov3_repo_dir = "dinov3"
dinov3_ckpt_path = "dinov3/dinov3_vitb16_pretrain.pth"
dinov3_embed_dim = 768
view_order = ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT","CAM_BACK","CAM_BACK_LEFT"]
depth = 19
depth_single_blocks = 38
cross_view_list = list(range(depth,depth_single_blocks,9))
is_multi_view = True
mv_order_map = {
    0: [5, 1],
    1: [0, 2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4, 0],
}
num_round = 1
# 1-GPU settings: match training resolution (112x200), short clip, fewer steps.
# Model 22.5GB bf16 + activations fits in one 47GB card with grad checkpointing.
num_frames = 9
height = 112
width = 208  # must give even latent width (208/8=26); 200 gives odd 25 vs z=26 mismatch
start_index = 0
end_index = 20000000000  # full val set (same as previous infer.py setting)
# Dataset settings
dataset = dict(
    type="nuscenes_video",
    pkl_path = "data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl",
    transform_name="resize_crop",
    fps_max=24,
    vmaf=False,
    memory_efficient=False,
    view_order=view_order,
    use_depth=use_depth,
    use_seg=use_seg,
    traj_ctrl=traj_ctrl,
    max_depth = 100,
    num_frames=num_frames,
    height=height,
    width=width,
    is_train=False,
)

grad_ckpt_settings = (100, 100)  # set the grad checkpoint settings

condition_config = dict(
    i2v_head=5,  # train i2v (image as first frame) with weight 5
)

if use_depth and use_seg:
    in_channels = 192
elif not use_depth and use_seg:
    in_channels = 128
elif use_depth and not use_seg:
    in_channels = 128
else:
    in_channels = 64

# Define model components
model = dict(
    type="flux",
    from_pretrained = "/data1/qjyuan/OmniNWM/outputs/stage3_anchor_kv_2gpu/260903_221844-train_stage_3_2gpu/epoch0-global_step5000",
    strict_load=False,
    fused_qkv=False,
    use_liger_rope=True,
    grad_ckpt_settings=grad_ckpt_settings,
    # model architecture
    in_channels=in_channels,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=depth,
    depth_single_blocks=depth_single_blocks,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    mv_order_map=mv_order_map,
    cross_view_list=cross_view_list,
    use_depth = use_depth,
    use_seg = use_seg,
    use_multi_level_noise = use_multi_level_noise,
    use_reg = use_reg,
    reg_align_layer = 8,
    dinov3_embed_dim = dinov3_embed_dim,
    reg_loss_weight = 0.1,
    # Anchor KV Cache 配置
    use_anchor_cache = True,
    anchor_cache_layer = 0,
    anchor_inject_double_layers = [4, 8, 12, 16],
    anchor_inject_single_layers = [5, 15, 25, 35],
)

ae = dict(
    type="hunyuan_vae",
    from_pretrained="pretrained/hunyuan_vae.safetensors",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,
    use_temporal_tiling=False,
)

is_causal_vae = True
# Optimization settings (unused in inference)
lr = 1e-4
eps = 1e-15

optim = dict(
    cls="HybridAdam",
    lr=lr,
    eps=eps,
    weight_decay=0.0,
    adamw_mode=True,
)

warmup_steps = 0
update_warmup_steps = True

grad_clip = 1.0
accumulation_steps = 1
ema_decay = None

# Acceleration settings: single GPU, no sequence parallelism, no booster.
prefetch_factor = 4
num_workers = 2
num_bucket_build_workers = 4

dtype = "bf16"
plugin = "zero2"
plugin_config = dict()
grad_checkpoint = True

async_io = False


sampling_option = dict(
    num_frames=num_frames,  # number of frames
    num_steps=25,  # reduced from 50 to speed up 1-GPU inference
    shift=True,
    temporal_reduction=4,
    is_causal_vae=True,
    method="i2v",  # hard-coded for now
    seed=None,  # random seed for z
    width=width,
    height=height,
    num_round = num_round,
)


# Other settings
seed = 42
outputs = "./outputs/infer_stage3_1gpu_reg1"
save_dir = "./outputs/infer_stage3_1gpu_reg1"
epochs = 10000
log_every = 1
ckpt_every = 100
keep_n_latest = 200000
wandb_project = "omninwm"

save_master_weights = True
load_master_weights = True
load = None
start_from_scratch = True

# eval
save_for_eval = True

enable_cache: True
