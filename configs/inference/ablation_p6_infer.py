# Inference config for ablation P6 (Anchor KV Cache) on PandaSet
# Based on ablation_infer.py protocol: 13 frames, 224x400, 50 steps, 200 val samples.
# NOTE vs scripts/pandaset_ablation_eval.sh: that script's sed misses depth_seg_root
# (pandaset depth/seg must come from precomputed files, unlike nuscenes path substitution).

calculate_reward = False
use_multi_level_noise = True
use_low_men_vae_infer = True
traj_ctrl = True
use_depth = True
use_seg = True

# === P6: frame-aligned REG + Anchor KV Cache (same as training) ===
use_reg = True
reg_use_align_loss = True
reg_frame_aligned = True
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
num_frames = 13
height = 224
width = 400
start_index = 0
end_index = 200  # Evaluate 200 samples (same as a-series protocol)
num_samples = 1

# Dataset settings — PandaSet validation split with precomputed depth/seg
dataset = dict(
    type="nuscenes_video",
    pkl_path = "/data1/qjyuan/OmniNWM/data/pandaset_pkl/pandaset_interp_12Hz_infos_val.pkl",
    dataset_name = "pandaset",
    depth_seg_root = "/data1/qjyuan/OmniNWM/data/pandaset_depth_seg",
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

grad_ckpt_settings = (100, 100)

condition_config = dict(
    i2v_head=5,
)

in_channels = 192  # depth + seg

model = dict(
    type="flux",
    # P6 latest checkpoint (epoch1, optimizer step 3000 ≈ 6000 data steps)
    from_pretrained = "/data1/qjyuan/OmniNWM/outputs/ablation_pandaset/p6_anchor_cache/260906_170609-train_ablation_p6_anchor_cache_pandaset/epoch1-global_step3000",
    strict_load=False,
    fused_qkv=False,
    use_liger_rope=True,
    grad_ckpt_settings=grad_ckpt_settings,
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
    use_depth=use_depth,
    use_seg=use_seg,
    use_multi_level_noise=use_multi_level_noise,
    use_reg=use_reg,
    reg_align_layer=8,
    reg_use_align_loss=reg_use_align_loss,
    reg_frame_aligned=reg_frame_aligned,
    dinov3_embed_dim=dinov3_embed_dim,
    reg_loss_weight=0.1,
    # ===== Anchor KV Cache: MUST match training config exactly =====
    # (gates keyed by layer index in ParameterDict; downsample/tokens are behavioral)
    use_anchor_cache=True,
    anchor_cache_layer=0,
    anchor_inject_double_layers=[4, 8, 12, 16],
    anchor_inject_single_layers=[5, 15, 25, 35],
    anchor_downsample=8,   # default is 8, explicit for clarity
    anchor_tokens=128,     # NOTE: model config default is 256 — training used 128!
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
accumulation_steps = 1
ema_decay = None

prefetch_factor = 4
num_workers = 2
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

sampling_option = dict(
    num_frames=num_frames,
    num_steps=50,
    shift=True,
    temporal_reduction=4,
    is_causal_vae=True,
    method="i2v",
    seed=None,
    width=width,
    height=height,
    num_round=num_round,
)

seed = 42
outputs = "./outputs/ablation_pandaset/p6_anchor_cache_eval"
save_dir = "./outputs/ablation_pandaset/p6_anchor_cache_eval"

epochs = 10000
log_every = 1
ckpt_every = 100
keep_n_latest = 200000
wandb_project = "omninwm_ablation"

save_master_weights = True
load_master_weights = True
load = None
start_from_scratch = True

# Save tensors for evaluation
save_for_eval = True
batch_size = 1
