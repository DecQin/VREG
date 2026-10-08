from IPython import embed
import gc
import math
import os
import random
import subprocess
import warnings
from contextlib import nullcontext
from copy import deepcopy
from pprint import pformat
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
gc.disable()
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
    unpack
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


@torch.no_grad()
def extract_dinov3_features(video_tensor, model_dinov3, chunk_size=8):
    # DINOv3 may live on CPU (to save GPU memory); move to GPU temporarily.
    was_cpu = next(model_dinov3.parameters()).device.type == "cpu"
    if was_cpu:
        model_dinov3 = model_dinov3.to(video_tensor.device, non_blocking=True)
    try:
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
    finally:
        if was_cpu:
            model_dinov3 = model_dinov3.to("cpu", non_blocking=True)
    return cls_tokens


def build_dinov3(cfg, device, dtype):
    import sys
    dinov3_repo_dir = cfg.get("dinov3_repo_dir", "dinov3")
    if dinov3_repo_dir not in sys.path:
        sys.path.insert(0, dinov3_repo_dir)
    from dinov3.models.vision_transformer import DinoVisionTransformer, vit_base
    dinov3_ckpt_path = cfg.get("dinov3_ckpt_path", None)
    model_dinov3 = vit_base(patch_size=16)  # , embed_dim=768, depth=12, num_heads=12, ffn_ratio=4
    if dinov3_ckpt_path is not None:
        ckpt = torch.load(dinov3_ckpt_path, map_location='cpu')
        if 'model' in ckpt:
            ckpt = ckpt['model']
        model_dinov3.load_state_dict(ckpt, strict=False)
    model_dinov3 = model_dinov3.to(device, dtype).eval().requires_grad_(False)
    return model_dinov3

def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs()

    # == get dtype & device ==
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32   = False
    device, coordinator = setup_device()
    grad_ckpt_buffer_size = cfg.get("grad_ckpt_buffer_size", 0)
    if grad_ckpt_buffer_size > 0:
        GLOBAL_ACTIVATION_MANAGER.setup_buffer(grad_ckpt_buffer_size, dtype)
    checkpoint_io = CheckpointIO()
    set_seed(cfg.get("seed", 1024))
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
        cfg.get("outputs", "./outputs"),
        model_name=config_to_name(cfg),
        config=cfg.to_dict(),
        exp_name=cfg.get("exp_name", None),  # useful for automatic restart to specify the exp_name
    )

    if is_log_process(plugin_type, plugin_config):
        print(f"changing {exp_dir} to share")
        os.system(f"chgrp -R share {exp_dir}")

    # == init logger, tensorboard & wandb ==
    logger = create_logger(exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    tb_writer = None
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(
                project=cfg.get("wandb_project", "Open-Sora"),
                name=exp_name,
                config=cfg.to_dict(),
                dir=exp_dir,
            )

    num_gpus = dist.get_world_size() if dist.is_initialized() else 1
    tp_size = cfg["plugin_config"].get("tp_size", 1)
    sp_size = cfg["plugin_config"].get("sp_size", 1)
    pp_size = cfg["plugin_config"].get("pp_size", 1)
    num_groups = num_gpus // (tp_size * sp_size * pp_size)
    logger.info("Number of GPUs: %s", num_gpus)
    logger.info("Number of groups: %s", num_groups)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    cache_pin_memory = pin_memory_cache_pre_alloc_numels is not None
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
        cache_pin_memory=cache_pin_memory,
        num_groups=num_groups,
    )
    print_mem("before prepare_dataloader")
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    print_mem("after prepare_dataloader")
    num_steps_per_epoch = len(dataloader)


    # ======================================================
    # 4. build model
    # ======================================================
    logger.info("Building models...")
    model = build_module(cfg.model, MODELS, device_map=device, torch_dtype=dtype).train()
    if cfg.get("grad_checkpoint", True):
        set_grad_checkpoint(model)
    log_cuda_memory("diffusion")
    log_model_params(model)

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

    model_dinov3 = None
    if cfg.get("use_reg", False) and (cfg.get("reg_use_depth_cls", True) or cfg.get("reg_use_seg_cls", True)):
        # Keep DINOv3 on CPU (~0.2GB): 2-GPU run is ~90MB short of VRAM at optimizer
        # step. extract_dinov3_features moves it to GPU temporarily per call.
        dinov3_device = "cpu" if cfg.get("dinov3_on_cpu", False) else device
        model_dinov3 = build_dinov3(cfg, dinov3_device, dtype)
        logger.info(f"DINOv3 model built for REG (device={dinov3_device}).")

    # == setup optimizer ==
    optimizer = create_optimizer(model, cfg.optim)

    # == setup lr scheduler ==
    lr_scheduler = create_lr_scheduler(
        optimizer=optimizer,
        num_steps_per_epoch=num_steps_per_epoch,
        epochs=cfg.get("epochs", 1000),
        warmup_steps=cfg.get("warmup_steps", None),
        use_cosine_scheduler=cfg.get("use_cosine_scheduler", False),
    )

    log_cuda_memory("optimizer")

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    logger.info("Preparing for distributed training...")

    # == boosting ==
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

    # == resume ==
    load_master_weights = cfg.get("load_master_weights", False)
    save_master_weights = cfg.get("save_master_weights", False)
    start_epoch = cfg.get("start_epoch", None)
    start_step = cfg.get("start_step", None)

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
        # load optimizer and scheduler will overwrite some of the hyperparameters, so we need to reset them
        set_lr(optimizer, lr_scheduler, cfg.optim.lr, cfg.get("initial_lr", None))
        if cfg.optim.get("eps", None) is not None:
            set_eps(optimizer, cfg.optim.eps)

        if cfg.get("update_warmup_steps", False):
            assert (
                cfg.get("warmup_steps", None) is not None
            ), "you need to set warmup_steps in order to pass --update-warmup-steps True"
            # set_warmup_steps(lr_scheduler, cfg.warmup_steps)
            lr_scheduler.step(start_epoch * num_steps_per_epoch + start_step)
            logger.info("The learning rate starts from %s", optimizer.param_groups[0]["lr"])

    if start_step is not None:
        # if start step exceeds data length, go to next epoch
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

    # == sharding EMA model ==
    if ema is not None:
        model_sharding(ema)
        ema = ema.to(device)
        log_cuda_memory("sharding EMA")

    # =======================================================
    # 5. training iter
    # =======================================================
    sigma_min = cfg.get("sigma_min", 1e-5)
    accumulation_steps = cfg.get("accumulation_steps", 1)
    ckpt_every = cfg.get("ckpt_every", 0)

    def _encode_views_chunked(x, chunk=2):
        # VAE-encode views in small chunks to cap the encoder activation peak
        # at multi-frame training (num_frames>=5). Exact: the causal 3D VAE
        # processes batch entries independently, so chunking the batch (view)
        # dim is lossless — only the transient memory footprint shrinks.
        if x.shape[0] <= chunk:
            return model_ae.encode(x)
        return torch.cat(
            [model_ae.encode(x[i : i + chunk]) for i in range(0, x.shape[0], chunk)], dim=0
        )

    @torch.no_grad()
    def prepare_inputs(batch):
        inp = dict()
        x = batch["video"]
        ray_map_x = batch.get("ray_map",None)
        depth_x = batch.get("depth",None)
        seg_x = batch.get("seg",None)
        ori_bs = x.shape[0]
        if cfg.get("is_multi_view",False):
            x = rearrange(x,'b n c t h w -> (b n) c t h w')
            if seg_x is not None:
                seg_x = rearrange(seg_x,'b n c t h w -> (b n) c t h w')
            if depth_x is not None:
                depth_x = rearrange(depth_x,'b n c t h w -> (b n) c t h w')
            if ray_map_x is not None:
                ray_map_x = rearrange(ray_map_x,'b n c t h w -> (b n) c t h w')
        bs = x.shape[0]

        num_cam = bs//ori_bs

        # == encode video ==
        with nsys.range("encode_video"), timers["encode_video"]:
            # == prepare condition ==
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
                depth_x_0 = _encode_views_chunked(depth_x)
                x_0 = torch.cat([x_0, depth_x_0], dim=1)
                if cfg.get("use_reg", False) and cfg.get("reg_use_depth_cls", True):
                    depth_cls_0 = extract_dinov3_features(depth_x, model_dinov3)
                del depth_x, depth_x_0

            if seg_x is not None:
                seg_x_0 = _encode_views_chunked(seg_x)
                x_0 = torch.cat([x_0, seg_x_0], dim=1)
                if cfg.get("use_reg", False) and cfg.get("reg_use_seg_cls", True):
                    seg_cls_0 = extract_dinov3_features(seg_x, model_dinov3)
                del seg_x, seg_x_0

        # == prepare timestep ==
        # follow SD3 time shift, shift_alpha = 1 for 256px and shift_alpha = 3 for 1024px
        shift_alpha = get_res_lin_function()((x_0.shape[-1] * x_0.shape[-2]) // 4)
        # add temporal influence
        shift_alpha *= math.sqrt(x_0.shape[-3])  # for image, T=1 so no effect
        if cfg.get("use_multi_level_noise",False):
            t_bs,t_c,t_num_frames,t_height,t_width = x_0.shape
            t = torch.sigmoid(torch.randn((t_bs,t_num_frames), device=device))
        else:
            t = torch.sigmoid(torch.randn((ori_bs), device=device))

        t = time_shift(shift_alpha, t).to(dtype)
        if not cfg.get("use_multi_level_noise",False):
            t = repeat(t,'b ... -> (b t) ...',t=num_cam)

        # == encode text ==
        with nsys.range("encode_text"), timers["encode_text"]:
                inp_ = prepare(
                    x_0,
                    ray_map_x,
                    patch_size=cfg.get("patch_size", 2),
                )
                inp.update(inp_)
                inp['x_ori'] = x_0

        t_rev = 1 - t
        if cfg.get("use_multi_level_noise",False):
            x_1 = torch.randn_like(x_0, dtype=torch.float32).to(device, dtype)
            x_t = t_rev[:,None, :, None, None] * x_0 + (1 - (1 - sigma_min) * t_rev[:,None, :, None, None]) * x_1
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
        with nsys.range("forward"), timers["forward"]:
            model_pred = model(**inp)
            v_t = (1 - sigma_min) * x_1 - x_0

            if cfg.get("use_reg", False):
                # Model may return (img, align_loss) or just img
                if isinstance(model_pred, (tuple, list)):
                    img_pred, align_loss = model_pred[0], model_pred[1]
                else:
                    img_pred = model_pred
                    align_loss = None
                loss_main = F.mse_loss(img_pred.float(), v_t.float(), reduction="mean")
                loss = loss_main
                align_loss_val = 0.0

                if align_loss is not None:
                    # Progressive REG loss warmup - start small, increase slowly
                    reg_weight = cfg.get("reg_loss_weight", 0.1)  # 从较小的值开始
                    if cfg.get("reg_loss_warmup", True):
                        warmup_steps = cfg.get("reg_loss_warmup_steps", 5000)
                        max_reg_weight = cfg.get("reg_loss_weight_max", 0.3)
                        # Global step for warmup (across epochs)
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
        # == backward & update ==
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
                # Free cached memory before optimizer step to avoid OOM
                torch.cuda.empty_cache()
                # NOTE: removed booster.checkpoint_io.synchronize() — it can pile up
                # CPU-side tensors before optimizer.step, causing CPU OOM on 6 GPU (192GB).
                optimizer.step()
                optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

        # == update EMA ==
        if ema is not None:
            with nsys.range("update_ema"), timers["update_ema"]:
                update_ema(
                    ema,
                    model.unwrap(),
                    optimizer=optimizer,
                    decay=cfg.get("ema_decay", 0.9999),
                )
        # torch.cuda.empty_cache()
        return loss_item, loss_main_item, align_loss_val
    
    # =======================================================
    # 6. training loop
    # =======================================================
    dist.barrier()
    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)

        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not is_log_process(plugin_type, plugin_config),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            pbar_iter = iter(pbar)
            # prefetch one for non-blocking data loading
            def fetch_data():
                step, batch = next(pbar_iter)
                pinned_video = batch["video"]
                batch["video"] = pinned_video.to(device, dtype, non_blocking=True)
                if batch.get('depth',None) is not None:
                    pinned_depth = batch["depth"]
                    batch["depth"] = pinned_depth.to(device, dtype, non_blocking=True)
                if batch.get('seg',None) is not None:
                    pinned_seg_video = batch["seg"]
                    batch["seg"] = pinned_seg_video.to(device, dtype, non_blocking=True)
                if batch.get('ray_map',None) is not None:
                    pinned_ray_map = batch["ray_map"]
                    batch["ray_map"] = pinned_ray_map.to(device, dtype, non_blocking=True)

                return batch, step, pinned_video
            
            batch_, step_, pinned_video_ = fetch_data()

            for _ in range(start_step, num_steps_per_epoch):
                nsys.step()
                # == load data ===
                with nsys.range("load_data"), timers["load_data"]:
                    batch, step, pinned_video = batch_, step_, pinned_video_
                    if step + 1 < num_steps_per_epoch:
                        # only fetch new data if not last step
                        batch_, step_, pinned_video_ = fetch_data()
                # == run iter ==
                with nsys.range("iter"), timers["iter"]:
                    inp, x_0, x_1, depth_cls_0, seg_cls_0 = prepare_inputs(batch)
                    if cache_pin_memory:
                        dataloader_iter.remove_cache(pinned_video)
                    loss, loss_main, loss_align = run_iter(inp, x_0, x_1, depth_cls_0, seg_cls_0)

                # == update log info ==
                if loss is not None:
                    running_loss += loss
                    running_loss_main += loss_main
                    running_loss_align += loss_align

                # == debug memory growth (temporary) ==
                if step < 5 or step % 50 == 0:
                    print_mem(f"after step {step}")

                # == log config ==
                global_step = epoch * num_steps_per_epoch + step
                actual_update_step = (global_step + 1) // accumulation_steps
                log_step += 1
                acc_step += 1

                # == logging ==
                if (global_step + 1) % accumulation_steps == 0:
                    if actual_update_step % cfg.get("log_every", 1) == 0:
                        if is_log_process(plugin_type, plugin_config):
                            avg_loss = running_loss / log_step
                            avg_loss_main = running_loss_main / log_step
                            avg_loss_align = running_loss_align / log_step
                            # progress bar
                            pbar.set_postfix(
                                {
                                    "loss": avg_loss,
                                    "main": avg_loss_main,
                                    "align": avg_loss_align,
                                    "global_grad_norm": optimizer.get_grad_norm(),
                                    "step": step,
                                    "global_step": global_step,
                                    "lr": optimizer.param_groups[0]["lr"],
                                }
                            )
                            # tensorboard
                            if tb_writer is not None:
                                tb_writer.add_scalar("loss", loss, actual_update_step)
                                tb_writer.add_scalar("loss_main", loss_main, actual_update_step)
                                tb_writer.add_scalar("loss_align", loss_align, actual_update_step)
                            # wandb
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
                                    "eps": optimizer.param_groups[0]["eps"],
                                    "global_grad_norm": optimizer.get_grad_norm(),
                                }
                                if cfg.get("record_time", False):
                                    wandb_dict.update(timers.to_dict())
                                wandb.log(wandb_dict, step=actual_update_step)

                        running_loss = 0.0
                        running_loss_main = 0.0
                        running_loss_align = 0.0
                        log_step = 0

                # == checkpoint saving ==
                # uncomment below 3 lines to forcely clean cache
                with nsys.range("clean_cache"), timers["clean_cache"]:
                    # Per-step gc + empty_cache to prevent CPU memory growth under ZeRO2+cpu_offload
                    gc.collect()
                    torch.cuda.empty_cache()
                    if ckpt_every > 0 and actual_update_step > 0 and actual_update_step % ckpt_every == 0 and coordinator.is_master():
                        subprocess.run("echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true", shell=True)

                with nsys.range("checkpoint"), timers["checkpoint"]:
                    if ckpt_every > 0 and actual_update_step > 0 and actual_update_step % ckpt_every == 0:
                        # mannual garbage collection
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

                        logger.info(
                            "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                            epoch,
                            step + 1,
                            actual_update_step,
                            save_dir,
                        )

                        # remove old checkpoints
                        rm_checkpoints(exp_dir, keep_n_latest=cfg.get("keep_n_latest", -1))
                        logger.info("Removed old checkpoints and kept %s latest ones.", cfg.get("keep_n_latest", -1))
                # uncomment below 3 lines to benchmark checkpoint
                # if ckpt_every > 0 and actual_update_step % ckpt_every == 0:
                #     booster.checkpoint_io._sync_io()
                #     checkpoint_io._sync_io()
                # == terminal timer ==
                if cfg.get("record_time", False):
                    print(timers.to_str(epoch, step))

        sampler.reset()
        start_step = 0
    log_cuda_max_memory("final")

if __name__ == "__main__":
    main()
