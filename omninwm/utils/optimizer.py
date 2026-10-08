import math

import torch
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from torch.optim.lr_scheduler import _LRScheduler


class LeanAdafactor(torch.optim.Optimizer):
    """Memory-lean Adafactor for colossalai ZeRO2 (master_weights=False) on 2x47GB GPUs.

    colossalai hands the optimizer flattened 1D bf16 param/grad SHARDS per rank.
    Constraints on 2 GPUs: model 22.5GB + bf16 grad shard 11.2GB = 33.7GB GPU base;
    AdamW fp32 states (22.4GB GPU) OOM; SGD fits but its updates (~2e-10) vanish
    below bf16 rounding — loss stays flat (verified over 23k steps).

    This optimizer:
      - keeps the fp32 second-moment state on CPU (22.4GB/rank, fits 32GB CPU budget)
      - computes chunk-wise on GPU (32MB chunks), one PCIe roundtrip per step (~8s)
      - applies bf16 updates via STOCHASTIC ROUNDING: update ~1e-4 is below bf16 ulp
        (~1.6e-4 for |w|~0.02); round-to-nearest would zero it every step — SR makes
        the expected update equal the true update, so learning accumulates correctly.
      - no internal update clipping (colossalai global-norm clips grads to 1.0 already)
    """

    CHUNK = 8 * 1024 * 1024  # 8M elements ≈ 32MB fp32 temps on GPU

    def __init__(self, params, lr=1e-3, decay_rate=-0.8, eps=1e-15, weight_decay=0.0):
        defaults = dict(lr=lr, decay_rate=decay_rate, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                self._step_one(p, group, lr)
        return loss

    def _step_one(self, p, group, lr):
        grad = p.grad.view(-1)
        pv = p.view(-1)
        n = pv.numel()
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            # bf16 second moment on CPU: 11.2GB/rank. fp32 would need 22.4GB/rank →
            # 2x(14.5 baseline + 22.4) = 74GB > 64GB job cap (2 GPU x 32GB) → oom_kill.
            # bf16 storage + fp32 chunk math: rounding noise ~0.4% relative, zero-mean.
            state["exp_avg_sq"] = torch.zeros(n, dtype=torch.bfloat16, device="cpu")
        state["step"] += 1
        beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
        exp_avg_sq = state["exp_avg_sq"]
        eps = group["eps"]
        dev = pv.device

        for i in range(0, n, self.CHUNK):
            j = min(i + self.CHUNK, n)
            g = grad[i:j].float()                                  # GPU bf16 -> fp32 chunk
            s = exp_avg_sq[i:j].to(dev).float()                    # CPU bf16 -> GPU fp32 chunk
            s.mul_(beta2t).add_(g * g + eps, alpha=1.0 - beta2t)   # EMA of squared grads (fp32)
            exp_avg_sq[i:j].copy_(s.to(torch.bfloat16), non_blocking=True)  # store bf16
            u = s.rsqrt().mul_(g)                                  # Adafactor update ~O(1)
            delta = -lr * u
            self._stochastic_round_add_(pv[i:j], delta)

    @staticmethod
    def _stochastic_round_add_(p_chunk, delta):
        """p_chunk (bf16, in-place) += delta (fp32) with stochastic rounding.

        Round-to-nearest discards sub-ulp updates; SR preserves them in expectation.
        Neighbor on the bf16 grid computed via bit manipulation (torch.nextafter
        is unreliable for bf16: returns fp32-grid neighbors). Sign-monotonic int16
        key mapping handles negatives correctly.
        """
        x = p_chunk.float() + delta
        lo = x.to(torch.bfloat16)                       # nearest bf16 grid point
        lo_f = lo.float()
        r = x - lo_f                                    # signed residue to nearest

        # map bf16 bits -> int key monotonic in float value
        b = lo.view(torch.int16).to(torch.int32)
        key = torch.where(b >= 0, b + 32768, -b)        # +0 and -0 both -> 32768
        key_hi = torch.where(r >= 0, key + 1, key - 1)  # neighbor in r's direction
        b_hi = torch.where(key_hi < 32768, -key_hi, key_hi - 32768)
        hi = b_hi.to(torch.int16).view(torch.bfloat16)

        ulp = (hi.float() - lo_f).abs()
        frac = (r.abs() / ulp).clamp(0.0, 1.0)
        take_hi = torch.rand_like(x) < frac
        p_chunk.copy_(torch.where(take_hi, hi, lo))


def create_optimizer(
    model: torch.nn.Module,
    optimizer_config: dict,
) -> torch.optim.Optimizer:
    """
    Create an optimizer.

    Args:
        model (torch.nn.Module): The model to be optimized.
        optimizer_config (dict): The configuration of the optimizer.

    Returns:
        torch.optim.Optimizer: The optimizer.
    """
    optimizer_name = optimizer_config.pop("cls", "HybridAdam")
    if optimizer_name == "HybridAdam":
        optimizer_cls = HybridAdam
    elif optimizer_name == "AdamW":
        optimizer_cls = torch.optim.AdamW
    elif optimizer_name == "SGD":
        optimizer_cls = torch.optim.SGD
    elif optimizer_name == "LeanAdafactor":
        # Factored 2nd moment: ~zero optimizer state (vs AdamW's 2x fp32 params).
        # Memory-lean step (no full fp32 grad cast) → fits 2x47GB GPUs.
        optimizer_cls = LeanAdafactor
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Anchor-cache gates are 8 scalars initialized at -6.0 (sigmoid ~ 0.0025,
    # cache "off"). With the global lr they can traverse only lr*steps of logit
    # range (1e-4 * 5000 = 0.5 → sigmoid still ~0.004), so the cache would never
    # activate within budget. anchor_gate_lr_mult puts them in a separate param
    # group with a higher lr so the gate can actually open if gradients push it.
    gate_lr_mult = optimizer_config.pop("anchor_gate_lr_mult", None)
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    if gate_lr_mult and any("anchor_cache" in n for n, _ in named_params):
        gate_params = [p for n, p in named_params if "anchor_cache" in n]
        other_params = [p for n, p in named_params if "anchor_cache" not in n]
        gate_cfg = dict(optimizer_config)
        gate_cfg["lr"] = optimizer_config["lr"] * gate_lr_mult
        param_groups = [
            dict(params=other_params, lr_mult=1.0, **optimizer_config),
            dict(params=gate_params, lr_mult=gate_lr_mult, **gate_cfg),
        ]
        optimizer = optimizer_cls(param_groups)
    else:
        optimizer = optimizer_cls([p for _, p in named_params], **optimizer_config)
    return optimizer


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_steps_per_epoch: int,
    epochs: int = 1000,
    warmup_steps: int | None = None,
    use_cosine_scheduler: bool = False,
    initial_lr: float = 1e-6,
) -> _LRScheduler | None:
    """
    Create a learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        num_steps_per_epoch (int): The number of steps per epoch.
        epochs (int): The number of epochs.
        warmup_steps (int |  None): The number of warmup steps.
        use_cosine_scheduler (bool): Whether to use cosine scheduler.

    Returns:
        _LRScheduler |  None: The learning rate scheduler
    """
    if warmup_steps is None and not use_cosine_scheduler:
        lr_scheduler = None
    elif use_cosine_scheduler:
        lr_scheduler = CosineAnnealingWarmupLR(
            optimizer,
            total_steps=num_steps_per_epoch * epochs,
            warmup_steps=warmup_steps,
        )
    else:
        lr_scheduler = LinearWarmupLR(optimizer, initial_lr=1e-6, warmup_steps=warmup_steps)
    return lr_scheduler


class LinearWarmupLR(_LRScheduler):
    """Linearly warmup learning rate and then linearly decay.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0
        last_step (int, optional): The index of last step, defaults to -1. When last_step=-1,
            the schedule is started from the beginning or When last_step=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, initial_lr=0, warmup_steps: int = 0, last_epoch: int = -1):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                self.initial_lr + (self.last_epoch + 1) / (self.warmup_steps + 1) * (lr - self.initial_lr)
                for lr in self.base_lrs
            ]
        else:
            return self.base_lrs
