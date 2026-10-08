# Ablation P7: Multi-frame training (13 frames) + Anchor KV Cache on PandaSet — 4 GPU
#
# ROOT-CAUSE FIX for "KVCache didn't improve FVD / later frames degrade".
# The entire P0-P6 series fine-tuned at num_frames=1, which:
#   (1) Makes the anchor cache degenerate: with T=1 the anchor IS the only
#       frame, so inject() is self-to-self downsampled cross-attention — pure
#       redundancy with the block's own self-attention. Gradients push all 8
#       gates closed (observed sigmoid 0.0002-0.0029 at step 10000, even more
#       closed than step 7500) → at inference the cache contributes ~nothing.
#   (2) Erodes the base model's temporal attention prior (the FVD killer):
#       10-20k steps of single-frame fine-tune at lr=1e-4 re-purposes temporal
#       attention for image generation; later frames of a 13-frame rollout
#       degrade while frame 0 stays clean only because it is the i2v condition.
#   (3) Never exercises the i2v_head condition: T=1 takes the image branch of
#       prepare_visual_condition_causal, so cond channels are all-zero during
#       training, while inference feeds mask=1 + GT first-frame latent.
# P7 trains on 13-frame clips @ 224x400 — exactly the evaluation protocol
# (13 frames, 224x400, i2v first-frame) — with gates initialized at 0.0
# (sigmoid 0.5) so the cache participates from step one and gradients decide.
#
# vs P6 (the only intended differences):
#   - num_frames: 1 -> 13 (bucket_config + video_attr_list match, required to
#     avoid the StopIteration empty-dataloader trap)
#   - anchor_gate_init: -6.0 (hardcoded before) -> 0.0 via new config field
#   - init from the same clean baseline as P0-P6 (NOT the eroded P6 ckpt)
# Companion control: p7b_multiframe_nocache_pandaset_4gpu.py — identical
# protocol with use_anchor_cache=False. Run both to attribute FVD gains to
# multi-frame training per se vs the cache mechanism.
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
    # ===== Anchor KV Cache, INSPATIO-style historical cache =====
    # P8 = Step 1 lightweight variant of full autoregressive chunk-by-chunk
    # (Step 2-4). Cache holds 3 frames (anchor + 2 history), each contributed
    # via a separate extract_and_store call at anchor_layer=0. Inject layers
    # cross-attend to the concatenation.
    use_anchor_cache=True,
    anchor_cache_layer=0,                    # extract anchor from double block 0
    anchor_inject_double_layers=[4, 8, 12, 16],
    anchor_inject_single_layers=[5, 15, 25, 35],
    anchor_downsample=8,                      # 350 tokens/frame -> ~44 anchors/frame
    anchor_tokens=128,
    anchor_gate_init=0.0,                     # sigmoid 0.5: active from step one
    anchor_history_frames=[0, 1, 2],          # INSPATIO historical: 3 frames concatenated
    # ===================================================
    reg_use_align_loss=reg_use_align_loss,
    reg_frame_aligned=reg_frame_aligned,
)

is_causal_vae = True
lr = 1e-4
eps = 1e-15

# AdamW as in P5/P6. Multi-frame raises activation memory ~13x but grad
# checkpointing keeps the per-block peak modest; fixed zero2 costs stay
# ~39.3GB/rank (22.5 model + 5.6 grad shard + 11.2 fetch) < 47GB.
optim = dict(
    cls="AdamW",
    lr=lr,
    eps=eps,
    weight_decay=0.0,
    # Gates start at 0.0; 100x lr lets them traverse the full sigmoid range in
    # ~600 steps so the cache can be shaped (open or closed) early in training.
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
outputs = "./outputs/ablation_pandaset/p7_anchor_cache_multiframe_4gpu"
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
