"""
Convergence comparison training script.
Based on train_reg.py with additions:
  - Timing/profiling (step_time, elapsed_time, gpu_hours, peak_memory)
  - Metrics logging to JSONL (train_metrics.jsonl, timing_metrics.jsonl)
  - Fairness checks at startup
  - run_metadata.json with variant/seed/param counts
  - RNG state save/restore for exact resume
  - Convergence-specific checkpoint saving at designated steps
"""
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import time
import warnings
from contextlib import nullcontext
from copy import deepcopy
from pprint import pformat

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
gc.disable()

import numpy as np
import torchvision.transforms as transforms
from einops import rearrange, repeat
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from colossalai.booster import Booster
from colossalai.utils import set_seed
from tqdm import tqdm

from omninwm.acceleration.checkpoint import (
    GLOBAL_ACTIVATION_MANAGER,
    set_grad_checkpoint,
)
from omninwm.acceleration.parallel_states import get_data_parallel_group
from omninwm.datasets.aspect import bucket_to_shapes
from omninwm.datasets.dataloader import prepare_dataloader
from omninwm.datasets.pin_memory_cache import PinMemoryCache
from omninwm.models.mmdit.distributed import MMDiTPolicy
from omninwm.registry import DATASETS, MODELS, build_module
from omninwm.utils.ckpt import (
    CheckpointIO,
    model_sharding,
    record_model_param_shape,
    rm_checkpoints,
)
from omninwm.utils.config import (
    config_to_name,
    create_experiment_workspace,
    parse_configs,
)
from omninwm.utils.logger import create_logger
from omninwm.utils.misc import (
    NsysProfiler,
    Timers,
    all_reduce_mean,
    create_tensorboard_writer,
    is_log_process,
    log_cuda_max_memory,
    log_cuda_memory,
    log_model_params,
    print_mem,
    to_torch_dtype,
)
from omninwm.utils.optimizer import create_lr_scheduler, create_optimizer
from omninwm.utils.sampling import (
    get_res_lin_function,
    pack,
    prepare,
    time_shift,
    unpack,
)
from omninwm.utils.train import (
    create_colossalai_plugin,
    prepare_visual_condition_causal,
    set_eps,
    set_lr,
    setup_device,
    update_ema,
)

torch.backends.cudnn.benchmark = False


# ─── DINOv3 helpers (same as train_reg.py) ───

@torch.no_grad()
def extract_dinov3_features(video_tensor, model_dinov3, chunk_size=8):
    B, C, T, H, W = video_tensor.shape
    frames = rearrange(video_tensor, 'b c t h w -> (b t) c h w')
    frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)
    frames = (frames + 1) / 2
    frames = frames.clamp(0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    cls_tokens_list = []
    for i in range(0, frames.shape[0], chunk_size):
        chunk = frames[i:i + chunk_size]
        with torch.no_grad():
            cls_tokens_list.append(model_dinov3(chunk))
    cls_tokens = torch.cat(cls_tokens_list, dim=0)
    cls_tokens = rearrange(cls_tokens, '(b t) d -> b t d', b=B, t=T)
    return cls_tokens


def build_dinov3(cfg, device, dtype):
    import sys
    dinov3_repo_dir = cfg.get("dinov3_repo_dir", "dinov3")
    if dinov3_repo_dir not in sys.path:
        sys.path.insert(0, dinov3_repo_dir)
    from dinov3.models.vision_transformer import DinoVisionTransformer, vit_base
    dinov3_ckpt_path = cfg.get("dinov3_ckpt_path", None)
    model_dinov3 = vit_base(patch_size=16)
    if dinov3_ckpt_path is not None:
        ckpt = torch.load(dinov3_ckpt_path, map_location='cpu')
        if 'model' in ckpt:
            ckpt = ckpt['model']
        model_dinov3.load_state_dict(ckpt, strict=False)
    model_dinov3 = model_dinov3.to(device, dtype).eval().requires_grad_(False)
    return model_dinov3


# ─── Convergence-specific utilities ───

def set_seed_all(seed):
    """Set seed for all RNG sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rng_states():
    """Capture all RNG states for checkpoint."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())],
    }


def set_rng_states(states):
    """Restore all RNG states from checkpoint."""
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch_cpu" in states:
        torch.random.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states:
        for i, s in enumerate(states["torch_cuda"]):
            if i < torch.cuda.device_count():
                torch.cuda.set_rng_state(s, i)


def file_hash(path, algo="md5"):
    """Compute hash of a file."""
    if path is None or not os.path.isfile(path):
        return None
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def count_parameters(model):
    """Count total, trainable, and REG-specific parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reg_params = 0
    for name, p in model.named_parameters():
        if any(k in name for k in ["depth_cls_in", "seg_cls_in", "reg_proj"]):
            reg_params += p.numel()
    return total, trainable, reg_params


def write_jsonl(path, record):
    """Append one JSON record to a JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def fairness_check(cfg, logger):
    """Verify critical config fields are present and consistent."""
    required_keys = ["seed", "lr", "accumulation_steps", "batch_size", "dtype"]
    warnings_list = []
    for k in required_keys:
        if cfg.get(k, None) is None:
            warnings_list.append(f"Missing critical config key: {k}")

    # Check base checkpoint exists
    load_path = cfg.get("load", None)
    if load_path and not os.path.isdir(load_path):
        warnings_list.append(f"Base checkpoint not found: {load_path}")

    if warnings_list:
        for w in warnings_list:
            logger.warning("[FAIRNESS] %s", w)
    return warnings_list


def save_run_metadata(exp_dir, cfg, total_params, trainable_params, reg_params,
                      base_ckpt_hash, num_gpus):
    """Save run_metadata.json."""
    metadata = {
        "variant": cfg.get("convergence_variant", "unknown"),
        "seed": cfg.get("seed", 42),
        "base_checkpoint": cfg.get("load", None),
        "base_checkpoint_hash": base_ckpt_hash,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "reg_extra_parameters": reg_params,
        "effective_batch_size": cfg.get("batch_size", None) * cfg.get("accumulation_steps", 1) * num_gpus,
        "num_gpus": num_gpus,
        "lr": cfg.get("lr", None) or cfg.get("optim", {}).get("lr", None),
        "accumulation_steps": cfg.get("accumulation_steps", 1),
        "use_reg": cfg.get("use_reg", False),
        "reg_frame_aligned": cfg.get("reg_frame_aligned", False),
        "reg_frame_injection": cfg.get("reg_frame_injection", True),
        "reg_use_align_loss": cfg.get("reg_use_align_loss", True),
        "config_hash": hashlib.md5(
            json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode()
        ).hexdigest()[:12],
    }
    # GPU / CUDA info
    try:
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    metadata["cuda_version"] = torch.version.cuda
    metadata["pytorch_version"] = torch.__version__

    path = os.path.join(exp_dir, "run_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


# ─── Main training function ───

def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    cfg = parse_configs()

    # == Convergence variant ==
    variant = cfg.get("convergence_variant", "unknown")
    seed = cfg.get("seed", 42)

    # == Set all seeds early ==
    set_seed_all(seed)

    # == get dtype & device ==
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device, coordinator = setup_device()
    grad_ckpt_buffer_size = cfg.get("grad_ckpt_buffer_size", 0)
    if grad_ckpt_buffer_size > 0:
        GLOBAL_ACTIVATION_MANAGER.setup_buffer(grad_ckpt_buffer_size, dtype)
    checkpoint_io = CheckpointIO()
    set_seed(seed)
    PinMemoryCache.force_dtype = dtype
    pin_memory_cache_pre_alloc_numels = cfg.get("pin_memory_cache_pre_alloc_numels", None)
    PinMemoryCache.pre_alloc_numels = pin_memory_cache_pre_alloc_numels

    # == init ColossalAI booster ==
    plugin_type = cfg.get("plugin", "zero2")
    plugin_config = cfg.get("plugin_config", {})
    plugin_kwargs = {}
    if plugin_type == "hybrid":
        plugin_kwargs["custom_policy"] = MMDiTPolicy
    plugin = create_colossalai_plugin(
        plugin=plugin_type,
        dtype=cfg.get("dtype", "bf16"),
        grad_clip=cfg.get("grad_clip", 0),
        **plugin_config,
        **plugin_kwargs,
    )
    booster = Booster(plugin=plugin)
    seq_align = plugin_config.get("sp_size", 1)

    # == init exp_dir ==
    exp_name, exp_dir = create_experiment_workspace(
        cfg.get("outputs", "./outputs/convergence"),
        model_name=config_to_name(cfg),
        config=cfg.to_dict(),
        exp_name=cfg.get("exp_name", None),
    )

    if is_log_process(plugin_type, plugin_config):
        os.system(f"chgrp -R share {exp_dir}")

    # == init logger ==
    logger = create_logger(exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    logger.info("Convergence variant: %s, seed: %s", variant, seed)

    tb_writer = None
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(
                project=cfg.get("wandb_project", "omninwm_convergence"),
                name=f"{variant}_seed{seed}",
                config=cfg.to_dict(),
                dir=exp_dir,
            )

    # == Fairness check ==
    fairness_warnings = fairness_check(cfg, logger)
    if fairness_warnings:
        logger.warning("Fairness check found %d warnings!", len(fairness_warnings))

    num_gpus = dist.get_world_size() if dist.is_initialized() else 1
    tp_size = cfg["plugin_config"].get("tp_size", 1)
    sp_size = cfg["plugin_config"].get("sp_size", 1)
    pp_size = cfg["plugin_config"].get("pp_size", 1)
    num_groups = num_gpus // (tp_size * sp_size * pp_size)
    logger.info("Number of GPUs: %s", num_gpus)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))

    cache_pin_memory = pin_memory_cache_pre_alloc_numels is not None
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 4),
        seed=seed,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
        cache_pin_memory=cache_pin_memory,
        num_groups=num_groups,
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    model = build_module(cfg.model, MODELS, device_map=device, torch_dtype=dtype).train()
    if cfg.get("grad_checkpoint", True):
        set_grad_checkpoint(model)
    log_cuda_memory("diffusion")
    log_model_params(model)

    # Count parameters
    total_params, trainable_params, reg_params = count_parameters(model)
    logger.info("Total params: %s, Trainable: %s, REG extra: %s", total_params, trainable_params, reg_params)

    # == build EMA model ==
    if cfg.get("ema_decay", None) is not None:
        ema = deepcopy(model).cpu().eval().requires_grad_(False)
        ema_shape_dict = record_model_param_shape(ema)
        logger.info("EMA model created.")
    else:
        ema = ema_shape_dict = None
        logger.info("No EMA model created.")
    log_cuda_memory("EMA")

    model_ae = build_module(cfg.ae, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
    del model_ae.decoder
    log_cuda_memory("autoencoder")
    log_model_params(model_ae)
    model_ae.encode = torch.compile(model_ae.encoder, dynamic=True)

    # == DINOv3 (only for REG variants) ==
    model_dinov3 = None
    if cfg.get("use_reg", False):
        model_dinov3 = build_dinov3(cfg, device, dtype)
        logger.info("DINOv3 model built for REG.")
    else:
        logger.info("No DINOv3 (baseline mode).")

    # == setup optimizer ==
    optimizer = create_optimizer(model, cfg.optim)
    lr_scheduler = create_lr_scheduler(
        optimizer=optimizer,
        num_steps_per_epoch=num_steps_per_epoch,
        epochs=cfg.get("epochs", 1000),
        warmup_steps=cfg.get("warmup_steps", None),
        use_cosine_scheduler=cfg.get("use_cosine_scheduler", False),
    )
    log_cuda_memory("optimizer")

    # ======================================================
    # 4. distributed training preparation
    # ======================================================
    logger.info("Preparing for distributed training...")
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boosted model for distributed training")
    log_cuda_memory("boost")

    # == Save run metadata ==
    base_ckpt_hash = file_hash(
        os.path.join(cfg.get("load", ""), "model.safetensors")
        if cfg.get("load") else None
    )
    if coordinator.is_master():
        save_run_metadata(exp_dir, cfg, total_params, trainable_params, reg_params,
                          base_ckpt_hash, num_gpus)
        # Save resolved config as YAML
        import yaml
        with open(os.path.join(exp_dir, "resolved_config.yaml"), "w") as f:
            yaml.dump(cfg.to_dict(), f, default_flow_style=False)

    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    log_step = acc_step = 0
    running_loss = 0.0
    running_loss_main = 0.0
    running_loss_align = 0.0
    timers = Timers(record_time=cfg.get("record_time", False), record_barrier=cfg.get("record_barrier", False))
    nsys = NsysProfiler(
        warmup_steps=cfg.get("nsys_warmup_steps", 2),
        num_steps=cfg.get("nsys_num_steps", 2),
        enabled=cfg.get("nsys", False),
    )
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == Convergence timing state ==
    train_start_time = time.time()
    profiling_warmup_steps = cfg.get("profiling_warmup_steps", 20)
    profile_interval = cfg.get("profile_interval", 50)
    step_times = []

    # JSONL paths
    train_metrics_path = os.path.join(exp_dir, "train_metrics.jsonl")
    timing_metrics_path = os.path.join(exp_dir, "timing_metrics.jsonl")

    # == resume ==
    load_master_weights = cfg.get("load_master_weights", False)
    save_master_weights = cfg.get("save_master_weights", False)
    start_epoch = cfg.get("start_epoch", None)
    start_step = cfg.get("start_step", None)

    # RNG states for resume
    rng_states = None

    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint from %s", cfg.load)
        lr_scheduler_to_load = lr_scheduler
        if cfg.get("update_warmup_steps", False):
            lr_scheduler_to_load = None

        ret = checkpoint_io.load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=None if cfg.get("start_from_scratch", False) else optimizer,
            lr_scheduler=None if cfg.get("start_from_scratch", False) else lr_scheduler_to_load,
            sampler=None if cfg.get("start_from_scratch", False) else sampler,
            include_master_weights=load_master_weights,
        )
        start_epoch = start_epoch if start_epoch is not None else ret[0]
        start_step = start_step if start_step is not None else ret[1]
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, ret[0], ret[1])
        set_lr(optimizer, lr_scheduler, cfg.optim.lr, cfg.get("initial_lr", None))
        set_eps(optimizer, cfg.optim.eps)

        if cfg.get("update_warmup_steps", False):
            assert cfg.get("warmup_steps", None) is not None
            lr_scheduler.step(start_epoch * num_steps_per_epoch + start_step)
            logger.info("The learning rate starts from %s", optimizer.param_groups[0]["lr"])

        # Try to load RNG states
        rng_path = os.path.join(cfg.load, "rng_states.pt")
        if os.path.isfile(rng_path):
            rng_states = torch.load(rng_path, map_location="cpu")
            set_rng_states(rng_states)
            logger.info("Restored RNG states from checkpoint.")

        # Try to load elapsed time
        timing_path = os.path.join(cfg.load, "convergence_timing.json")
        if os.path.isfile(timing_path):
            with open(timing_path) as f:
                timing_data = json.load(f)
            train_start_time = time.time() - timing_data.get("elapsed_seconds", 0)
            logger.info("Restored elapsed time: %.1f sec", timing_data.get("elapsed_seconds", 0))

    if start_step is not None:
        if start_step > num_steps_per_epoch:
            start_epoch = (
                start_epoch + start_step // num_steps_per_epoch
                if start_epoch is not None
                else start_step // num_steps_per_epoch
            )
            start_step = start_step % num_steps_per_epoch
    else:
        start_step = 0
    sampler.set_step(start_step)
    start_epoch = start_epoch if start_epoch is not None else 0
    logger.info("Starting from epoch %s step %s", start_epoch, start_step)

    if ema is not None:
        model_sharding(ema)
        ema = ema.to(device)
        log_cuda_memory("sharding EMA")

    # ======================================================
    # 5. training iter
    # ======================================================
    sigma_min = cfg.get("sigma_min", 1e-5)
    accumulation_steps = cfg.get("accumulation_steps", 1)
    ckpt_every = cfg.get("ckpt_every", 0)
    convergence_save_steps = set(cfg.get("convergence_save_steps", []))

    @torch.no_grad()
    def prepare_inputs(batch):
        inp = dict()
        x = batch["video"]
        ray_map_x = batch.get("ray_map", None)
        depth_x = batch.get("depth", None)
        seg_x = batch.get("seg", None)
        ori_bs = x.shape[0]
        if cfg.get("is_multi_view", False):
            x = rearrange(x, 'b n c t h w -> (b n) c t h w')
            if seg_x is not None:
                seg_x = rearrange(seg_x, 'b n c t h w -> (b n) c t h w')
            if depth_x is not None:
                depth_x = rearrange(depth_x, 'b n c t h w -> (b n) c t h w')
            if ray_map_x is not None:
                ray_map_x = rearrange(ray_map_x, 'b n c t h w -> (b n) c t h w')
        bs = x.shape[0]
        num_cam = bs // ori_bs

        with nsys.range("encode_video"), timers["encode_video"]:
            if cfg.get("condition_config", None) is not None:
                x_0, cond = prepare_visual_condition_causal(x, cfg.condition_config, model_ae)
                cond = pack(cond, patch_size=cfg.get("patch_size", 2))
                inp["cond"] = cond
            else:
                x_0 = model_ae.encode(x)
            del x

            depth_cls_0 = None
            seg_cls_0 = None

            if depth_x is not None:
                depth_x_0 = model_ae.encode(depth_x)
                x_0 = torch.cat([x_0, depth_x_0], dim=1)
                if cfg.get("use_reg", False) and cfg.get("reg_use_depth_cls", True):
                    depth_cls_0 = extract_dinov3_features(depth_x, model_dinov3)
                del depth_x, depth_x_0

            if seg_x is not None:
                seg_x_0 = model_ae.encode(seg_x)
                x_0 = torch.cat([x_0, seg_x_0], dim=1)
                if cfg.get("use_reg", False) and cfg.get("reg_use_seg_cls", True):
                    seg_cls_0 = extract_dinov3_features(seg_x, model_dinov3)
                del seg_x, seg_x_0

        shift_alpha = get_res_lin_function()((x_0.shape[-1] * x_0.shape[-2]) // 4)
        shift_alpha *= math.sqrt(x_0.shape[-3])
        if cfg.get("use_multi_level_noise", False):
            t_bs, t_c, t_num_frames, t_height, t_width = x_0.shape
            t = torch.sigmoid(torch.randn((t_bs, t_num_frames), device=device))
        else:
            t = torch.sigmoid(torch.randn((ori_bs), device=device))

        t = time_shift(shift_alpha, t).to(dtype)
        if not cfg.get("use_multi_level_noise", False):
            t = repeat(t, 'b ... -> (b t) ...', t=num_cam)

        with nsys.range("encode_text"), timers["encode_text"]:
            inp_ = prepare(x_0, ray_map_x, patch_size=cfg.get("patch_size", 2))
            inp.update(inp_)
            inp['x_ori'] = x_0

        t_rev = 1 - t
        if cfg.get("use_multi_level_noise", False):
            x_1 = torch.randn_like(x_0, dtype=torch.float32).to(device, dtype)
            x_t = t_rev[:, None, :, None, None] * x_0 + (1 - (1 - sigma_min) * t_rev[:, None, :, None, None]) * x_1
            x_0 = pack(x_0, patch_size=cfg.get("patch_size", 2))
            x_1 = pack(x_1, patch_size=cfg.get("patch_size", 2))
            x_t = pack(x_t, patch_size=cfg.get("patch_size", 2))
            if cfg.get("use_reg", False):
                if depth_cls_0 is not None:
                    inp["depth_cls"] = depth_cls_0
                    inp["depth_cls_ref"] = depth_cls_0
                if seg_cls_0 is not None:
                    inp["seg_cls"] = seg_cls_0
                    inp["seg_cls_ref"] = seg_cls_0
        else:
            x_0 = pack(x_0, patch_size=cfg.get("patch_size", 2))
            x_1 = torch.randn_like(x_0, dtype=torch.float32).to(device, dtype)
            x_t = t_rev[:, None, None] * x_0 + (1 - (1 - sigma_min) * t_rev[:, None, None]) * x_1
            if cfg.get("use_reg", False):
                if depth_cls_0 is not None:
                    inp["depth_cls"] = depth_cls_0
                    inp["depth_cls_ref"] = depth_cls_0
                if seg_cls_0 is not None:
                    inp["seg_cls"] = seg_cls_0
                    inp["seg_cls_ref"] = seg_cls_0

        inp["img"] = x_t
        inp["timesteps"] = t.to(dtype)
        inp["guidance"] = torch.full((x_t.shape[0],), cfg.get("guidance", 4), device=x_t.device, dtype=x_t.dtype)
        return inp, x_0, x_1, depth_cls_0, seg_cls_0

    def run_iter(inp, x_0, x_1, depth_cls_0=None, seg_cls_0=None):
        step_start = time.time()

        with nsys.range("forward"), timers["forward"]:
            model_pred = model(**inp)
            v_t = (1 - sigma_min) * x_1 - x_0

            if cfg.get("use_reg", False):
                if isinstance(model_pred, (tuple, list)):
                    img_pred, align_loss = model_pred[0], model_pred[1]
                else:
                    img_pred = model_pred
                    align_loss = None
                loss_main = F.mse_loss(img_pred.float(), v_t.float(), reduction="mean")
                loss = loss_main
                align_loss_val = 0.0

                if align_loss is not None:
                    reg_weight = cfg.get("reg_loss_weight", 0.1)
                    if cfg.get("reg_loss_warmup", True):
                        warmup_steps = cfg.get("reg_loss_warmup_steps", 5000)
                        max_reg_weight = cfg.get("reg_loss_weight_max", 0.3)
                        current_step = epoch * num_steps_per_epoch + step
                        reg_weight = min(reg_weight * (current_step / warmup_steps), max_reg_weight)
                    loss = loss + reg_weight * align_loss
                    align_loss_val = align_loss.item()
            else:
                loss = F.mse_loss(model_pred.float(), v_t.float(), reduction="mean")
                loss_main = loss
                align_loss_val = 0.0

        loss_item = all_reduce_mean(loss.data.clone().detach()).item()
        loss_main_item = all_reduce_mean(loss_main.data.clone().detach()).item()

        dist.barrier()
        with nsys.range("backward"), timers["backward"]:
            ctx = (
                booster.no_sync(model, optimizer)
                if cfg.get("plugin", "zero2") in ("zero1", "zero1-seq") and (step + 1) % accumulation_steps != 0
                else nullcontext()
            )
            with ctx:
                booster.backward(loss=(loss / accumulation_steps), optimizer=optimizer)

        with nsys.range("optim"), timers["optim"]:
            if (step + 1) % accumulation_steps == 0:
                torch.cuda.empty_cache()
                booster.checkpoint_io.synchronize()
                optimizer.step()
                optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

        if ema is not None:
            with nsys.range("update_ema"), timers["update_ema"]:
                update_ema(ema, model.unwrap(), optimizer=optimizer, decay=cfg.get("ema_decay", 0.9999))

        step_time_sec = time.time() - step_start
        return loss_item, loss_main_item, align_loss_val, step_time_sec

    # ======================================================
    # 6. training loop
    # ======================================================
    dist.barrier()
    for epoch in range(start_epoch, cfg_epochs):
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)

        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not is_log_process(plugin_type, plugin_config),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            pbar_iter = iter(pbar)

            def fetch_data():
                step, batch = next(pbar_iter)
                pinned_video = batch["video"]
                batch["video"] = pinned_video.to(device, dtype, non_blocking=True)
                if batch.get('depth', None) is not None:
                    batch["depth"] = batch["depth"].to(device, dtype, non_blocking=True)
                if batch.get('seg', None) is not None:
                    batch["seg"] = batch["seg"].to(device, dtype, non_blocking=True)
                if batch.get('ray_map', None) is not None:
                    batch["ray_map"] = batch["ray_map"].to(device, dtype, non_blocking=True)
                return batch, step, pinned_video

            batch_, step_, pinned_video_ = fetch_data()

            for _ in range(start_step, num_steps_per_epoch):
                nsys.step()
                with nsys.range("load_data"), timers["load_data"]:
                    batch, step, pinned_video = batch_, step_, pinned_video_
                    if step + 1 < num_steps_per_epoch:
                        batch_, step_, pinned_video_ = fetch_data()

                with nsys.range("iter"), timers["iter"]:
                    inp, x_0, x_1, depth_cls_0, seg_cls_0 = prepare_inputs(batch)
                    if cache_pin_memory:
                        dataloader_iter.remove_cache(pinned_video)
                    loss, loss_main, loss_align, step_time_sec = run_iter(inp, x_0, x_1, depth_cls_0, seg_cls_0)

                if loss is not None:
                    running_loss += loss
                    running_loss_main += loss_main
                    running_loss_align += loss_align

                global_step = epoch * num_steps_per_epoch + step
                actual_update_step = (global_step + 1) // accumulation_steps
                log_step += 1
                acc_step += 1

                # Track step times (skip warmup)
                if global_step >= profiling_warmup_steps:
                    step_times.append(step_time_sec)

                # == Convergence timing metrics ==
                elapsed_sec = time.time() - train_start_time
                gpu_hours = elapsed_sec / 3600 * num_gpus

                # Write train_metrics.jsonl (rank 0 only)
                if coordinator.is_master() and (global_step + 1) % accumulation_steps == 0:
                    write_jsonl(train_metrics_path, {
                        "variant": variant,
                        "seed": seed,
                        "global_step": global_step,
                        "actual_update_step": actual_update_step,
                        "epoch": epoch,
                        "step_in_epoch": step,
                        "loss": loss,
                        "loss_main": loss_main,
                        "loss_align": loss_align,
                        "lr": optimizer.param_groups[0]["lr"],
                        "step_time_sec": step_time_sec,
                        "elapsed_sec": round(elapsed_sec, 2),
                        "gpu_hours": round(gpu_hours, 4),
                    })

                # Write timing_metrics.jsonl at profile_interval (rank 0 only)
                if coordinator.is_master() and global_step >= profiling_warmup_steps and global_step % profile_interval == 0:
                    torch.cuda.synchronize()
                    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
                    peak_reserved = torch.cuda.max_memory_reserved() / 1e9
                    avg_step_time = np.median(step_times[-50:]) if step_times else 0
                    batch_size_eff = cfg.get("batch_size", 1) * num_groups
                    samples_per_sec = batch_size_eff / avg_step_time if avg_step_time > 0 else 0

                    write_jsonl(timing_metrics_path, {
                        "variant": variant,
                        "seed": seed,
                        "global_step": global_step,
                        "step_time_sec": round(step_time_sec, 4),
                        "avg_step_time_sec": round(avg_step_time, 4),
                        "samples_per_sec": round(samples_per_sec, 2),
                        "elapsed_sec": round(elapsed_sec, 2),
                        "gpu_hours": round(gpu_hours, 4),
                        "peak_memory_allocated_gb": round(peak_alloc, 2),
                        "peak_memory_reserved_gb": round(peak_reserved, 2),
                        "loss": loss,
                        "loss_main": loss_main,
                        "loss_align": loss_align,
                    })
                    # Reset peak memory stats
                    torch.cuda.reset_peak_memory_stats()

                # == logging ==
                if (global_step + 1) % accumulation_steps == 0:
                    if actual_update_step % cfg.get("log_every", 1) == 0:
                        if is_log_process(plugin_type, plugin_config):
                            avg_loss = running_loss / log_step
                            avg_loss_main = running_loss_main / log_step
                            avg_loss_align = running_loss_align / log_step
                            pbar.set_postfix({
                                "loss": avg_loss,
                                "main": avg_loss_main,
                                "align": avg_loss_align,
                                "global_grad_norm": optimizer.get_grad_norm(),
                                "step": step,
                                "global_step": global_step,
                                "lr": optimizer.param_groups[0]["lr"],
                            })
                            if tb_writer is not None:
                                tb_writer.add_scalar("loss", loss, actual_update_step)
                                tb_writer.add_scalar("loss_main", loss_main, actual_update_step)
                                tb_writer.add_scalar("loss_align", loss_align, actual_update_step)
                            if cfg.get("wandb", False):
                                wandb_dict = {
                                    "iter": global_step,
                                    "acc_step": acc_step,
                                    "epoch": epoch,
                                    "loss": loss,
                                    "loss_main": loss_main,
                                    "loss_align": loss_align,
                                    "avg_loss": avg_loss,
                                    "avg_loss_main": avg_loss_main,
                                    "avg_loss_align": avg_loss_align,
                                    "lr": optimizer.param_groups[0]["lr"],
                                    "step_time_sec": step_time_sec,
                                    "elapsed_sec": elapsed_sec,
                                    "gpu_hours": gpu_hours,
                                    "global_grad_norm": optimizer.get_grad_norm(),
                                }
                                wandb.log(wandb_dict, step=actual_update_step)

                        running_loss = 0.0
                        running_loss_main = 0.0
                        running_loss_align = 0.0
                        log_step = 0

                # == checkpoint saving ==
                should_save = False
                if ckpt_every > 0 and actual_update_step % ckpt_every == 0:
                    should_save = True
                if (actual_update_step * accumulation_steps) in convergence_save_steps:
                    should_save = True

                if should_save:
                    with nsys.range("checkpoint"), timers["checkpoint"]:
                        gc.collect()
                        save_dir = checkpoint_io.save(
                            booster,
                            exp_dir,
                            model=model,
                            ema=ema,
                            optimizer=None,
                            lr_scheduler=lr_scheduler,
                            sampler=sampler,
                            epoch=epoch,
                            step=step + 1,
                            global_step=global_step + 1,
                            batch_size=cfg.get("batch_size", None),
                            actual_update_step=actual_update_step,
                            ema_shape_dict=ema_shape_dict,
                            async_io=cfg.get("async_io", False),
                            include_master_weights=save_master_weights,
                        )
                        if is_log_process(plugin_type, plugin_config):
                            os.system(f"chgrp -R share {save_dir}")

                        # Save RNG states
                        if coordinator.is_master():
                            rng_path = os.path.join(save_dir, "rng_states.pt")
                            torch.save(get_rng_states(), rng_path)

                            # Save elapsed time
                            timing_data = {
                                "elapsed_seconds": time.time() - train_start_time,
                                "global_step": global_step,
                                "variant": variant,
                            }
                            with open(os.path.join(save_dir, "convergence_timing.json"), "w") as f:
                                json.dump(timing_data, f, indent=2)

                        logger.info("Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                                    epoch, step + 1, actual_update_step, save_dir)
                        rm_checkpoints(exp_dir, keep_n_latest=cfg.get("keep_n_latest", -1))

        sampler.reset()
        start_step = 0

    log_cuda_max_memory("final")

    # == Final summary (rank 0 only) ==
    if coordinator.is_master():
        total_elapsed = time.time() - train_start_time
        total_gpu_hours = total_elapsed / 3600 * num_gpus
        avg_step = np.median(step_times) if step_times else 0
        summary = {
            "variant": variant,
            "seed": seed,
            "total_elapsed_sec": round(total_elapsed, 2),
            "total_gpu_hours": round(total_gpu_hours, 4),
            "avg_step_time_sec": round(avg_step, 4),
            "median_step_time_sec": round(np.median(step_times), 4) if step_times else 0,
            "total_steps": global_step + 1,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "reg_params": reg_params,
            "peak_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
        with open(os.path.join(exp_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Training summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
