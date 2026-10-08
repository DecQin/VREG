"""
Anchor KV Cache for OmniNWM

Inspired by INSPATIO-WORLD's Spatiotemporal Cache mechanism.
Adapted for OmniNWM's full-sequence generation paradigm:
- Instead of caching across autoregressive chunks, we cache the first frame's
  intermediate K/V as a spatial anchor, and inject it into subsequent frames
  at designated layers via cross-attention.
- This provides a persistent "spatial memory" that enhances temporal consistency
  across all generated frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class AnchorKVCache(nn.Module):
    """
    Frame-Anchored KV Cache module.

    At a designated anchor layer, extracts the first frame's intermediate
    features as K/V pairs. At subsequent injection layers, allows all frames
    to cross-attend to these anchor K/V via a gated residual connection.

    This ensures all generated frames maintain spatial consistency with the
    first frame (which has GT i2v conditioning).

    Memory optimization: anchor K/V are spatially downsampled via stride,
    reducing the cross-attention cost from O(T*H*W * H*W) to
    O(T*H*W * anchor_L). No extra learnable parameters beyond the gates.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        anchor_layer: int = 0,
        inject_double_layers: Optional[List[int]] = None,
        inject_single_layers: Optional[List[int]] = None,
        anchor_downsample: int = 16,
        anchor_tokens: int = 128,
        gate_init: float = -6.0,
        history_frames: Optional[List[int]] = None,
    ):
        """
        Args:
            hidden_size: Model hidden dimension (e.g. 3072)
            num_heads: Number of attention heads
            anchor_layer: Which double block layer to extract anchor K/V from
            inject_double_layers: Double block layer indices to inject anchor cache
            inject_single_layers: Single block layer indices to inject anchor cache
            anchor_downsample: Stride for downsampling anchor K/V tokens
            anchor_tokens: Target number of anchor tokens (fallback upper bound)
            gate_init: Initial logit value for gates. -6.0 → sigmoid ≈ 0.0025
                (cache nearly off, for loading pretrained ckpts without cache
                weights). 0.0 → sigmoid 0.5 (cache active from step one, for
                multi-frame training where the anchor carries real signal).
            history_frames: Which frame indices to extract KV from at anchor_layer.
                Default [0] = first-frame-only (P7 behavior). E.g. [0,1,2] =
                cache holds first 3 frames concatenated, INSPATIO-style historical
                cache. Larger sets expand memory linearly.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.anchor_layer = anchor_layer
        self.inject_double_layers = inject_double_layers or [8, 16]
        self.inject_single_layers = inject_single_layers or [15, 35]
        self.anchor_downsample = anchor_downsample
        self.anchor_tokens = anchor_tokens
        self.history_frames = history_frames if history_frames is not None else [0]

        # Learnable gates for each injection layer
        # gate_init=-6.0: sigmoid ≈ 0.0025 ≈ 0 (cache disabled by default, safe
        # when loading a pretrained ckpt that never trained the cache)
        # gate_init=0.0: sigmoid 0.5 (cache participates from the first step —
        # use with multi-frame training so gradients can shape it either way)
        self.double_gates = nn.ParameterDict({
            f"layer_{i}": nn.Parameter(torch.full((1,), gate_init))
            for i in self.inject_double_layers
        })
        self.single_gates = nn.ParameterDict({
            f"layer_{i}": nn.Parameter(torch.full((1,), gate_init))
            for i in self.inject_single_layers
        })

        # colossalai LowLevelZeroOptimizer escape hatch: params flagged with
        # skip_zero_check bypass the uniform-dtype assertion, so the fp32 gates
        # can live inside a bf16 model (see _apply for why fp32 is required).
        for pd in (self.double_gates, self.single_gates):
            for p in pd.values():
                p.skip_zero_check = True

        # Cache storage (populated during forward pass)
        self._double_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._single_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _apply(self, fn, recurse=True):
        # Keep gate parameters in fp32 despite module-wide bf16 casts (e.g. the
        # colossalai zero plugin converts via module.to(dtype), which lands here).
        # Rationale: bf16 storage at |gate|~6 has ULP = 0.03, ~300x larger than
        # AdamW's per-step update (lr=1e-4), so a bf16 gate at init -6.0 can never
        # move (observed: frozen at exactly -6.0000 through 5500 steps on both
        # LeanAdafactor and AdamW runs). fp32 gates take sub-ULP updates fine;
        # forward compute stays bf16 via the .to() cast in inject().
        super()._apply(fn, recurse)
        for pd in (self.double_gates, self.single_gates):
            for p in pd.values():
                if p.data.dtype != torch.float32:
                    p.data = p.data.float()
        return self

    def reset_cache(self):
        """Clear all cached K/V pairs."""
        self._double_cache.clear()
        self._single_cache.clear()

    def reset_cache_for_inject(self):
        """Same as reset_cache but kept for API symmetry."""
        self.reset_cache()

    def extract_and_store(
        self,
        img_hidden: torch.Tensor,
        img_ids_t: torch.Tensor,
        layer_idx: int,
        is_double: bool = True,
    ):
        """
        Extract first frame's features and store as downsampled K/V cache.

        Args:
            img_hidden: [B, T*H*W, hidden_size] intermediate features
            img_ids_t: [B, T*H*W] frame index for each token
            layer_idx: Current layer index
            is_double: Whether this is a double stream block
        """
        anchor_mask = (img_ids_t == 0)  # [B, T*H*W]

        # Extract first frame features per sample in batch
        anchor_feats_list = []
        for b in range(img_hidden.shape[0]):
            anchor_feats_list.append(img_hidden[b, anchor_mask[b]])  # [H*W, C]
        anchor_feats = torch.stack(anchor_feats_list, dim=0)  # [B, H*W, C]

        # Stride-based spatial downsample: take every d-th token
        # [B, H*W, C] -> [B, ceil(H*W/d), C]
        stride = self.anchor_downsample
        anchor_feats = anchor_feats[:, ::stride, :]

        # Further trim to anchor_tokens upper bound if needed
        if anchor_feats.shape[1] > self.anchor_tokens:
            anchor_feats = anchor_feats[:, :self.anchor_tokens, :]

        # Reshape to multi-head format: [B, num_heads, anchor_L, head_dim]
        B, L2, C = anchor_feats.shape
        K = anchor_feats.reshape(B, L2, self.num_heads, self.head_dim).transpose(1, 2)
        V = anchor_feats.reshape(B, L2, self.num_heads, self.head_dim).transpose(1, 2)

        # Detach to prevent gradient flow through cache
        cache = self._double_cache if is_double else self._single_cache
        cache[layer_idx] = (K.detach(), V.detach())

    def inject(
        self,
        img_hidden: torch.Tensor,
        layer_idx: int,
        is_double: bool = True,
        raymap_len: int = 0,
        chunk_size: int = 8192,
    ) -> torch.Tensor:
        """
        Inject anchor K/V into current frame features via chunked cross-attention.

        Args:
            img_hidden: [B, L, hidden_size] current features
                For double blocks: L = T*H*W (img tokens only)
                For single blocks: L = raymap_len + T*H*W (raymap + img tokens)
            layer_idx: Current layer index
            is_double: Whether this is a double stream block
            raymap_len: Number of raymap tokens (0 for double blocks)
            chunk_size: Max number of query tokens per attention chunk (memory saving)

        Returns:
            img_hidden with anchor cache injected (gated residual)
        """
        # NOTE: anchor K/V is extracted ONCE at self.anchor_layer (double block)
        # and shared by ALL injection layers (double and single). The old lookup
        # `layer_idx not in cache` never matched the extraction key, so injection
        # silently never fired and gates stayed at init forever.
        gates = self.double_gates if is_double else self.single_gates

        if self.anchor_layer not in self._double_cache or f"layer_{layer_idx}" not in gates:
            return img_hidden

        anchor_K, anchor_V = self._double_cache[self.anchor_layer]  # [B, num_heads, anchor_L, head_dim]
        gate = torch.sigmoid(gates[f"layer_{layer_idx}"])

        B, L, C = img_hidden.shape

        # For single blocks, only inject into img tokens (skip raymap tokens)
        if not is_double and raymap_len > 0:
            raymap_part = img_hidden[:, :raymap_len, :]
            img_part = img_hidden[:, raymap_len:, :]
        else:
            img_part = img_hidden

        # Current img features as Q: [B, num_heads, img_L, head_dim]
        img_L = img_part.shape[1]
        Q = img_part.reshape(B, img_L, self.num_heads, self.head_dim).transpose(1, 2)

        # Chunked cross-attention to avoid OOM on large sequences
        scale = self.head_dim ** -0.5
        anchor_out_chunks = []

        for start in range(0, img_L, chunk_size):
            end = min(start + chunk_size, img_L)
            Q_chunk = Q[:, :, start:end, :]  # [B, num_heads, chunk, head_dim]

            attn_weights = torch.matmul(Q_chunk, anchor_K.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights.float(), dim=-1).to(Q_chunk.dtype)
            anchor_out_chunk = torch.matmul(attn_weights, anchor_V)
            anchor_out_chunks.append(anchor_out_chunk)

        anchor_out = torch.cat(anchor_out_chunks, dim=2)  # [B, num_heads, img_L, head_dim]
        anchor_out = anchor_out.transpose(1, 2).reshape(B, img_L, C)

        # Gated residual (gate cast to hidden dtype so the residual stream
        # stays bf16; gradient still flows through the cast into the fp32 gate)
        img_part = img_part + gate.to(anchor_out.dtype) * anchor_out

        # Reconstruct full hidden if single block
        if not is_double and raymap_len > 0:
            img_hidden = torch.cat([raymap_part, img_part], dim=1)
        else:
            img_hidden = img_part

        return img_hidden

    def should_extract(self, layer_idx: int, is_double: bool = True) -> bool:
        """Check if this layer should extract anchor K/V."""
        return is_double and layer_idx == self.anchor_layer

    def should_inject(self, layer_idx: int, is_double: bool = True) -> bool:
        """Check if this layer should inject anchor K/V."""
        layers = self.inject_double_layers if is_double else self.inject_single_layers
        return layer_idx in layers
