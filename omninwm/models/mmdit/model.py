# # Modified from Flux
# #
# # Copyright 2024 Black Forest Labs

# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at

# #     http://www.apache.org/licenses/LICENSE-2.0

# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
# from einops import rearrange, repeat
# from dataclasses import dataclass, field
# import os
# import torch
# from torch import Tensor, nn
# from omninwm.acceleration.checkpoint import auto_grad_checkpoint
# from omninwm.models.mmdit.layers import (
#     DoubleStreamBlock,
#     EmbedND,
#     LastLayer,
#     LigerEmbedND,
#     MLPEmbedder,
#     SingleStreamBlock,
#     timestep_embedding,
#     pack,
# )
# from omninwm.registry import MODELS
# from omninwm.utils.ckpt import load_checkpoint
# import torch.nn.functional as F
# from omninwm.models.mmdit.anchor_cache import AnchorKVCache

# @dataclass
# class MMDiTConfig:
#     model_type = "MMDiT"
#     from_pretrained: str
#     cache_dir: str
#     in_channels: int
#     hidden_size: int
#     mlp_ratio: float
#     num_heads: int
#     depth: int
#     depth_single_blocks: int
#     axes_dim: list[int]
#     cross_view_list: list[int]
#     theta: int
#     qkv_bias: bool
#     fused_qkv: bool = True
#     grad_ckpt_settings: tuple[int, int] | None = None
#     use_liger_rope: bool = False
#     patch_size: int = 2
#     mv_order_map: dict = field(default_factory=dict)
#     use_depth: bool = False
#     use_seg: bool = False
#     use_multi_level_noise: bool = False
#     use_reg: bool = False
#     reg_align_layer: int = 8
#     dinov3_embed_dim: int = 768
#     reg_loss_weight: float = 1.0
#     reg_cross_attn_layers: list = field(default_factory=list)  # 向后兼容，不再使用
#     # Anchor KV Cache 参数
#     use_anchor_cache: bool = False
#     anchor_cache_layer: int = 0
#     anchor_inject_double_layers: list = field(default_factory=lambda: [4, 8, 12, 16])
#     anchor_inject_single_layers: list = field(default_factory=lambda: [5, 15, 25, 35])

#     def get(self, attribute_name, default=None):
#         return getattr(self, attribute_name, default)

#     def __contains__(self, attribute_name):
#         return hasattr(self, attribute_name)


# class MMDiTModel(nn.Module):
#     config_class = MMDiTConfig

#     def __init__(self, config: MMDiTConfig):
#         super().__init__()

#         self.config = config
#         self.in_channels = config.in_channels
#         self.out_channels = self.in_channels
#         self.patch_size = config.patch_size
#         self.mv_order_map = config.mv_order_map
#         self.cross_view_list = config.cross_view_list
#         self.use_depth = config.use_depth
#         self.use_seg = config.use_seg
#         self.use_multi_level_noise = config.use_multi_level_noise
#         self.use_reg = config.use_reg

#         if config.hidden_size % config.num_heads != 0:
#             raise ValueError(
#                 f"Hidden size {config.hidden_size} must be divisible by num_heads {config.num_heads}"
#             )

#         pe_dim = config.hidden_size // config.num_heads
#         if sum(config.axes_dim) != pe_dim:
#             raise ValueError(
#                 f"Got {config.axes_dim} but expected positional dim {pe_dim}"
#             )

#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_heads
#         pe_embedder_cls = LigerEmbedND if config.use_liger_rope else EmbedND
#         self.pe_embedder = pe_embedder_cls(
#             dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim
#         )

#         self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
#         self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)

#         if self.use_depth and self.use_seg:
#             cond_channels = self.in_channels // 3
#         elif self.use_depth or self.use_seg:
#             cond_channels = self.in_channels // 2
#         else:
#             cond_channels = self.in_channels

#         cond_input_dim = cond_channels + self.patch_size**2

#         self.cond_in = (
#             nn.Linear(cond_input_dim, self.hidden_size, bias=True)
#         )

#         self.double_blocks = nn.ModuleList(
#             [
#                 DoubleStreamBlock(
#                     self.hidden_size,
#                     self.num_heads,
#                     mlp_ratio=config.mlp_ratio,
#                     qkv_bias=config.qkv_bias,
#                     fused_qkv=config.fused_qkv,
#                     use_multi_level_noise = self.use_multi_level_noise
#                 )
#                 for _ in range(config.depth)
#             ]
#         )
#         self.single_blocks = nn.ModuleList(
#             [
#                 SingleStreamBlock(
#                     self.hidden_size,
#                     self.num_heads,
#                     mlp_ratio=config.mlp_ratio,
#                     fused_qkv=config.fused_qkv,
#                     mv_order_map=self.mv_order_map if layer_idx in self.cross_view_list else None,
#                     use_multi_level_noise = self.use_multi_level_noise
#                 )
#                 for layer_idx in range(config.depth_single_blocks)
#             ]
#         )
#         self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, use_multi_level_noise=self.use_multi_level_noise)
        
#         self.traj_cond_in = nn.Linear(24, self.hidden_size, bias=True)

#         if self.use_reg:
#             self.reg_align_layer = config.reg_align_layer
#             self.dinov3_embed_dim = config.dinov3_embed_dim
#             self.reg_loss_weight = config.reg_loss_weight
#             if self.use_depth:
#                 self.depth_cls_in = nn.Linear(config.dinov3_embed_dim, self.hidden_size, bias=True)
#             if self.use_seg:
#                 self.seg_cls_in = nn.Linear(config.dinov3_embed_dim, self.hidden_size, bias=True)
#             self.reg_proj = nn.Sequential(
#                 nn.Linear(self.hidden_size, self.hidden_size),
#                 nn.GELU(),
#                 nn.Linear(self.hidden_size, config.dinov3_embed_dim),
#             )

#         self.initialize_weights()

#         if self.config.grad_ckpt_settings:
#             self.forward = self.forward_selective_ckpt
#         else:
#             self.forward = self.forward_ckpt
#         self._input_requires_grad = False

#         # Anchor KV Cache
#         self.use_anchor_cache = getattr(config, 'use_anchor_cache', False)
#         if self.use_anchor_cache:
#             self.anchor_cache = AnchorKVCache(
#                 hidden_size=self.hidden_size,
#                 num_heads=self.num_heads,
#                 anchor_layer=getattr(config, 'anchor_cache_layer', 0),
#                 inject_double_layers=getattr(config, 'anchor_inject_double_layers', [4, 8, 12, 16]),
#                 inject_single_layers=getattr(config, 'anchor_inject_single_layers', [5, 15, 25, 35]),
#             )
            

#     def initialize_weights(self):
#         nn.init.zeros_(self.cond_in.weight)
#         nn.init.zeros_(self.cond_in.bias)
#         if self.use_reg:
#             if hasattr(self, 'depth_cls_in'):
#                 nn.init.zeros_(self.depth_cls_in.weight)
#                 nn.init.zeros_(self.depth_cls_in.bias)
#             if hasattr(self, 'seg_cls_in'):
#                 nn.init.zeros_(self.seg_cls_in.weight)
#                 nn.init.zeros_(self.seg_cls_in.bias)

#     def prepare_block_inputs(
#         self,
#         img: Tensor,
#         img_ids: Tensor,
#         raymap: Tensor,
#         raymap_ids: Tensor,
#         timesteps: Tensor,
#         cond: Tensor = None,
#         depth_cls: Tensor = None,
#         seg_cls: Tensor = None,
#     ):
#         img = self.img_in(img)
#         img = img + self.cond_in(cond)
        
#         if self.use_multi_level_noise:
#             if timesteps.dim() == 1:
#                 vec = self.time_in(timestep_embedding(timesteps, 256))
#                 vec = vec.unsqueeze(1)
#             else:
#                 timesteps_b, timesteps_t = timesteps.shape
#                 timesteps_rearrange = rearrange(
#                     timesteps, 'b t -> (b t)'
#                 )
#                 vec = self.time_in(timestep_embedding(timesteps_rearrange, 256))
#                 vec = rearrange(vec, '(b t) c -> b t c', b=timesteps_b, t=timesteps_t)
#         else:
#             vec = self.time_in(timestep_embedding(timesteps, 256))

#         raymap = pack(raymap, self.patch_size)
#         raymap = self.traj_cond_in(raymap)

#         ids = torch.cat((raymap_ids, img_ids), dim=1)
#         pe = self.pe_embedder(ids)

#         if self._input_requires_grad:
#             img.requires_grad_()
#             raymap.requires_grad_()

#         self._reg_context = None
#         if self.use_reg:
#             cls_embed_list = []
#             if depth_cls is not None and hasattr(self, 'depth_cls_in'):
#                 cls_embed_list.append(self.depth_cls_in(depth_cls))
#             if seg_cls is not None and hasattr(self, 'seg_cls_in'):
#                 cls_embed_list.append(self.seg_cls_in(seg_cls))
#             if cls_embed_list:
#                 # Frame-aligned REG injection: map CLS tokens to latent frames
#                 # and inject per-frame instead of global mean pooling.
#                 # This preserves temporal dynamics which is critical for moving viewpoints.
#                 cls_embed = torch.cat(cls_embed_list, dim=1)  # [B, T_cls*num_modalities, hidden_size]
#                 num_modalities = len(cls_embed_list)
#                 T_cls = cls_embed.shape[1] // num_modalities

#                 # Infer latent frame count from img_ids
#                 img_ids_t = img_ids[:, :, 0]  # [B, T*H*W]
#                 T_latent = int(img_ids_t.max().item()) + 1

#                 # Reshape cls_embed: [B, num_modalities, T_cls, hidden_size]
#                 cls_embed = rearrange(cls_embed, 'b (m t) c -> b m t c', m=num_modalities, t=T_cls)

#                 # Subsample T_cls to T_latent (uniform sampling)
#                 if T_cls >= T_latent:
#                     indices = torch.linspace(0, T_cls - 1, T_latent, device=cls_embed.device, dtype=torch.long)
#                     cls_embed = cls_embed[:, :, indices, :]  # [B, num_modalities, T_latent, hidden_size]
#                 else:
#                     pad = T_latent - T_cls
#                     cls_embed = torch.cat([cls_embed, cls_embed[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)

#                 # Average across modalities, keep frame dimension
#                 cls_embed = cls_embed.mean(dim=1)  # [B, T_latent, hidden_size]

#                 # Gather per-frame CLS embed for each img token based on its frame index
#                 frame_indices = img_ids_t.long()  # [B, T*H*W]
#                 cls_embed_per_token = cls_embed.gather(
#                     1, frame_indices.unsqueeze(-1).expand(-1, -1, cls_embed.shape[-1])
#                 )  # [B, T*H*W, hidden_size]

#                 img = img + cls_embed_per_token

#         return img, raymap, vec, pe

#     def enable_input_require_grads(self):
#         self._input_requires_grad = True

#     def compute_alignment_loss(self, img_hidden: Tensor, cls_ref: Tensor) -> Tensor:
#         img_pooled = img_hidden.mean(dim=1)
#         img_proj = self.reg_proj(img_pooled.float())
#         img_proj = F.normalize(img_proj, dim=-1)
#         cls_ref_pooled = cls_ref.mean(dim=1).float()
#         cls_ref_pooled = F.normalize(cls_ref_pooled, dim=-1)
#         cos_sim = torch.sum(img_proj * cls_ref_pooled.detach(), dim=-1)
#         loss = 2 - 2 * torch.mean(cos_sim)
#         return loss

#     def forward_ckpt(
#         self,
#         img: Tensor,
#         img_ids: Tensor,
#         raymap: Tensor,
#         raymap_ids: Tensor,
#         timesteps: Tensor,
#         cond: Tensor = None,
#         depth_cls: Tensor = None,
#         seg_cls: Tensor = None,
#         depth_cls_ref: Tensor = None,
#         seg_cls_ref: Tensor = None,
#         **kwargs,
#     ) -> Tensor:
#         img, raymap, vec, pe = self.prepare_block_inputs(
#             img, img_ids, raymap, raymap_ids, timesteps, cond, depth_cls, seg_cls
#         )
#         has_grad = img.grad_fn is not None

#         align_loss = None
#         img_ids_t = img_ids[:, :, 0]  # [B, T*H*W] frame indices

#         # Reset anchor cache at the start of each forward pass
#         if self.use_anchor_cache:
#             self.anchor_cache.reset_cache()

#         for i, block in enumerate(self.double_blocks):
#             if has_grad:
#                 img, raymap = auto_grad_checkpoint(block, img, raymap, vec, pe)
#             else:
#                 img, raymap = block(img, raymap, vec, pe)

#             # Anchor KV Cache: extract at anchor layer, inject at designated layers
#             if self.use_anchor_cache:
#                 if self.anchor_cache.should_extract(i, is_double=True):
#                     self.anchor_cache.extract_and_store(img, img_ids_t, i, is_double=True)
#                 if self.anchor_cache.should_inject(i, is_double=True):
#                     img = self.anchor_cache.inject(img, i, is_double=True)

#             if self.use_reg and i == self.reg_align_layer:
#                 cls_ref_list = []
#                 if depth_cls_ref is not None:
#                     cls_ref_list.append(depth_cls_ref)
#                 if seg_cls_ref is not None:
#                     cls_ref_list.append(seg_cls_ref)
#                 if cls_ref_list:
#                     cls_ref_cat = torch.cat(cls_ref_list, dim=1)
#                     align_loss = self.compute_alignment_loss(img, cls_ref_cat)

#         img = torch.cat((raymap, img), 1)
#         for i, block in enumerate(self.single_blocks):
#             if has_grad:
#                 img = auto_grad_checkpoint(block, img, vec, pe)
#             else:
#                 img = block(img, vec, pe)

#             # Anchor KV Cache injection in single blocks
#             if self.use_anchor_cache and self.anchor_cache.should_inject(i, is_double=False):
#                 # For single blocks, img contains both raymap and img tokens
#                 # We only inject into the img portion
#                 raymap_len = raymap.shape[1]
#                 img_tokens = img[:, raymap_len:, ...]
#                 img_tokens = self.anchor_cache.inject(img_tokens, i, is_double=False)
#                 img = torch.cat([img[:, :raymap_len, ...], img_tokens], dim=1)

#         raymap_len = raymap.shape[1]
#         img = img[:, raymap_len:, ...]

#         img = self.final_layer(img, vec)

#         if self.use_reg and align_loss is not None:
#             return img, align_loss
#         return img

#     def forward_selective_ckpt(
#         self,
#         img: Tensor,
#         img_ids: Tensor,
#         raymap: Tensor,
#         raymap_ids: Tensor,
#         timesteps: Tensor,
#         cond: Tensor = None,
#         depth_cls: Tensor = None,
#         seg_cls: Tensor = None,
#         depth_cls_ref: Tensor = None,
#         seg_cls_ref: Tensor = None,
#         **kwargs,
#     ) -> Tensor:

#         img, raymap, vec, pe = self.prepare_block_inputs(
#             img, img_ids, raymap, raymap_ids, timesteps, cond, depth_cls, seg_cls
#         )
#         has_grad = img.grad_fn is not None

#         align_loss = None
#         img_ids_t = img_ids[:, :, 0]  # [B, T*H*W] frame indices

#         # Reset anchor cache at the start of each forward pass
#         if self.use_anchor_cache:
#             self.anchor_cache.reset_cache()

#         ckpt_depth_double = self.config.grad_ckpt_settings[0]
#         for i, block in enumerate(self.double_blocks[:ckpt_depth_double]):
#             if has_grad:
#                 img, raymap = auto_grad_checkpoint(block, img, raymap, vec, pe)
#             else:
#                 img, raymap = block(img, raymap, vec, pe)

#             # Anchor KV Cache
#             if self.use_anchor_cache:
#                 if self.anchor_cache.should_extract(i, is_double=True):
#                     self.anchor_cache.extract_and_store(img, img_ids_t, i, is_double=True)
#                 if self.anchor_cache.should_inject(i, is_double=True):
#                     img = self.anchor_cache.inject(img, i, is_double=True)

#             if self.use_reg and i == self.reg_align_layer:
#                 cls_ref_list = []
#                 if depth_cls_ref is not None:
#                     cls_ref_list.append(depth_cls_ref)
#                 if seg_cls_ref is not None:
#                     cls_ref_list.append(seg_cls_ref)
#                 if cls_ref_list:
#                     cls_ref_cat = torch.cat(cls_ref_list, dim=1)
#                     align_loss = self.compute_alignment_loss(img, cls_ref_cat)

#         for i, block in enumerate(self.double_blocks[ckpt_depth_double:], start=ckpt_depth_double):
#             img, raymap = block(img, raymap, vec, pe)

#             # Anchor KV Cache
#             if self.use_anchor_cache:
#                 if self.anchor_cache.should_extract(i, is_double=True):
#                     self.anchor_cache.extract_and_store(img, img_ids_t, i, is_double=True)
#                 if self.anchor_cache.should_inject(i, is_double=True):
#                     img = self.anchor_cache.inject(img, i, is_double=True)

#             if self.use_reg and i == self.reg_align_layer:
#                 cls_ref_list = []
#                 if depth_cls_ref is not None:
#                     cls_ref_list.append(depth_cls_ref)
#                 if seg_cls_ref is not None:
#                     cls_ref_list.append(seg_cls_ref)
#                 if cls_ref_list:
#                     cls_ref_cat = torch.cat(cls_ref_list, dim=1)
#                     align_loss = self.compute_alignment_loss(img, cls_ref_cat)

#         ckpt_depth_single = self.config.grad_ckpt_settings[1]

#         img = torch.cat((raymap, img), 1)
#         for i, block in enumerate(self.single_blocks[:ckpt_depth_single]):
#             if has_grad:
#                 img = auto_grad_checkpoint(block, img, vec, pe)
#             else:
#                 img = block(img, vec, pe)

#             # Anchor KV Cache injection in single blocks
#             if self.use_anchor_cache and self.anchor_cache.should_inject(i, is_double=False):
#                 raymap_len = raymap.shape[1]
#                 img_tokens = img[:, raymap_len:, ...]
#                 img_tokens = self.anchor_cache.inject(img_tokens, i, is_double=False)
#                 img = torch.cat([img[:, :raymap_len, ...], img_tokens], dim=1)

#         for i, block in enumerate(self.single_blocks[ckpt_depth_single:], start=ckpt_depth_single):
#             img = block(img, vec, pe)

#             # Anchor KV Cache injection in single blocks
#             if self.use_anchor_cache and self.anchor_cache.should_inject(i, is_double=False):
#                 raymap_len = raymap.shape[1]
#                 img_tokens = img[:, raymap_len:, ...]
#                 img_tokens = self.anchor_cache.inject(img_tokens, i, is_double=False)
#                 img = torch.cat([img[:, :raymap_len, ...], img_tokens], dim=1)

#         raymap_len = raymap.shape[1]
#         img = img[:, raymap_len:, ...]

#         img = self.final_layer(img, vec)

#         if self.use_reg and align_loss is not None:
#             return img, align_loss
#         return img


# @MODELS.register_module("flux")
# def Flux(
#     cache_dir: str = None,
#     from_pretrained: str = None,
#     device_map: str | torch.device = "cuda",
#     torch_dtype: torch.dtype = torch.bfloat16,
#     strict_load: bool = False,
#     **kwargs,
# ) -> MMDiTModel:
#     config = MMDiTConfig(
#         from_pretrained=from_pretrained,
#         cache_dir=cache_dir,
#         **kwargs,
#     )
#     low_precision_init = from_pretrained is not None and len(from_pretrained) > 0
#     if low_precision_init:
#         default_dtype = torch.get_default_dtype()
#         torch.set_default_dtype(torch_dtype)
#     with torch.device(device_map):
#         model = MMDiTModel(config)
#     if low_precision_init:
#         torch.set_default_dtype(default_dtype)
#     else:
#         model = model.to(torch_dtype)
#     if from_pretrained:
#         # When transitioning from stage1 (in_channels=64) to stage2 (in_channels=192),
#         # strict_load=False would skip img_in entirely, losing pretrained RGB weights.
#         # Instead, we load the checkpoint manually and expand img_in weights.
#         ckpt_path = from_pretrained
#         if not os.path.exists(ckpt_path):
#             from omninwm.utils.ckpt import load_from_hf_hub
#             ckpt_path = load_from_hf_hub(ckpt_path, cache_dir)

#         if ckpt_path.endswith(".safetensors"):
#             from safetensors.torch import load_file
#             ckpt_state = load_file(ckpt_path, device=torch.cuda.current_device())
#         elif ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
#             ckpt_state = torch.load(ckpt_path, map_location=device_map)
#         else:
#             # Sharded checkpoint: use standard loading
#             model = load_checkpoint(
#                 model, from_pretrained, cache_dir=cache_dir,
#                 device_map=device_map, strict=strict_load,
#             )
#             ckpt_state = None

#         if ckpt_state is not None:
#             model_sd = model.state_dict()
#             # Expand img_in weights: [3072, 64] -> [3072, 192]
#             if 'img_in.weight' in ckpt_state and 'img_in.weight' in model_sd:
#                 ckpt_shape = ckpt_state['img_in.weight'].shape
#                 model_shape = model_sd['img_in.weight'].shape
#                 if ckpt_shape != model_shape and ckpt_shape[0] == model_shape[0]:
#                     old_w = ckpt_state['img_in.weight']
#                     new_w = torch.zeros(model_shape, dtype=old_w.dtype, device=old_w.device)
#                     new_w[:, :ckpt_shape[1]] = old_w
#                     ckpt_state['img_in.weight'] = new_w

#             # Expand final_layer.linear weights: [256, 3072] -> [768, 3072]
#             if 'final_layer.linear.weight' in ckpt_state and 'final_layer.linear.weight' in model_sd:
#                 ckpt_shape = ckpt_state['final_layer.linear.weight'].shape
#                 model_shape = model_sd['final_layer.linear.weight'].shape
#                 if ckpt_shape != model_shape and ckpt_shape[1] == model_shape[1]:
#                     old_w = ckpt_state['final_layer.linear.weight']
#                     new_w = torch.zeros(model_shape, dtype=old_w.dtype, device=old_w.device)
#                     new_w[:ckpt_shape[0]] = old_w
#                     ckpt_state['final_layer.linear.weight'] = new_w
#                     # Expand bias too
#                     if 'final_layer.linear.bias' in ckpt_state:
#                         old_b = ckpt_state['final_layer.linear.bias']
#                         new_b = torch.zeros(model_shape[0], dtype=old_b.dtype, device=old_b.device)
#                         new_b[:ckpt_shape[0]] = old_b
#                         ckpt_state['final_layer.linear.bias'] = new_b

#             # Load with strict=False to allow missing/new keys (e.g., depth_cls_in, seg_cls_in)
#             missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
#             from omninwm.utils.ckpt import print_load_warning
#             print_load_warning(missing, unexpected)
#     return model

# Modified from Flux
#
# Copyright 2024 Black Forest Labs

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from einops import rearrange, repeat
from dataclasses import dataclass, field
import os
import torch
from torch import Tensor, nn
from omninwm.acceleration.checkpoint import auto_grad_checkpoint
from omninwm.models.mmdit.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    LigerEmbedND,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
    pack,
)
from omninwm.registry import MODELS
from omninwm.utils.ckpt import load_checkpoint
import torch.nn.functional as F

@dataclass
class MMDiTConfig:
    model_type = "MMDiT"
    from_pretrained: str
    cache_dir: str
    in_channels: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    cross_view_list: list[int]
    theta: int
    qkv_bias: bool
    fused_qkv: bool = True
    grad_ckpt_settings: tuple[int, int] | None = None
    use_liger_rope: bool = False
    patch_size: int = 2
    mv_order_map: dict = field(default_factory=dict)
    use_depth: bool = False
    use_seg: bool = False
    use_multi_level_noise: bool = False
    use_reg: bool = False
    reg_align_layer: int = 8
    reg_use_align_loss: bool = True  # Ablation: disable alignment loss while keeping CLS injection
    dinov3_embed_dim: int = 768
    reg_loss_weight: float = 1.0
    reg_pred_weight: float = 0.1
    reg_frame_aligned: bool = False  # Frame-aligned alignment loss (preserves frame/view dims)
    reg_frame_injection: bool = True  # True=per-frame CLS injection (frame_reg), False=global broadcast (category_reg)
    reg_cross_attn_layers: list = field(default_factory=list)  # 向后兼容，不再使用
    # Anchor KV Cache 参数
    use_anchor_cache: bool = False
    anchor_cache_layer: int = 0
    anchor_inject_double_layers: list = field(default_factory=lambda: [4, 8, 12, 16])
    anchor_inject_single_layers: list = field(default_factory=lambda: [5, 15, 25, 35])
    anchor_downsample: int = 8
    anchor_tokens: int = 256
    anchor_gate_init: float = -6.0  # gate logit init; 0.0 → sigmoid 0.5 for multi-frame training
    anchor_history_frames: list = field(default_factory=lambda: [0])  # INSPATIO-style historical cache; [0]=first-frame-only

    def get(self, attribute_name, default=None):
        return getattr(self, attribute_name, default)

    def __contains__(self, attribute_name):
        return hasattr(self, attribute_name)


class MMDiTModel(nn.Module):
    config_class = MMDiTConfig

    def __init__(self, config: MMDiTConfig):
        super().__init__()

        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = self.in_channels
        self.patch_size = config.patch_size
        self.mv_order_map = config.mv_order_map
        self.cross_view_list = config.cross_view_list
        self.use_depth = config.use_depth
        self.use_seg = config.use_seg
        self.use_multi_level_noise = config.use_multi_level_noise
        self.use_reg = config.use_reg

        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by num_heads {config.num_heads}"
            )

        pe_dim = config.hidden_size // config.num_heads
        if sum(config.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {config.axes_dim} but expected positional dim {pe_dim}"
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        pe_embedder_cls = LigerEmbedND if config.use_liger_rope else EmbedND
        self.pe_embedder = pe_embedder_cls(
            dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim
        )

        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)

        if self.use_depth and self.use_seg:
            cond_channels = self.in_channels // 3
        elif self.use_depth or self.use_seg:
            cond_channels = self.in_channels // 2
        else:
            cond_channels = self.in_channels

        cond_input_dim = cond_channels + self.patch_size**2

        self.cond_in = (
            nn.Linear(cond_input_dim, self.hidden_size, bias=True)
        )

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    fused_qkv=config.fused_qkv,
                    use_multi_level_noise = self.use_multi_level_noise
                )
                for _ in range(config.depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    fused_qkv=config.fused_qkv,
                    mv_order_map=self.mv_order_map if layer_idx in self.cross_view_list else None,
                    use_multi_level_noise = self.use_multi_level_noise
                )
                for layer_idx in range(config.depth_single_blocks)
            ]
        )
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, use_multi_level_noise=self.use_multi_level_noise)
        
        self.traj_cond_in = nn.Linear(24, self.hidden_size, bias=True)

        if self.use_reg:
            self.reg_align_layer = config.reg_align_layer
            self.reg_use_align_loss = config.reg_use_align_loss
            self.dinov3_embed_dim = config.dinov3_embed_dim
            self.reg_loss_weight = config.reg_loss_weight
            if self.use_depth:
                self.depth_cls_in = nn.Linear(config.dinov3_embed_dim, self.hidden_size, bias=True)
            if self.use_seg:
                self.seg_cls_in = nn.Linear(config.dinov3_embed_dim, self.hidden_size, bias=True)
            self.reg_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, config.dinov3_embed_dim),
            )

        # Anchor KV Cache initialization
        self.use_anchor_cache = config.use_anchor_cache
        if self.use_anchor_cache:
            from omninwm.models.mmdit.anchor_cache import AnchorKVCache
            self.anchor_cache = AnchorKVCache(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                anchor_layer=config.anchor_cache_layer,
                inject_double_layers=config.anchor_inject_double_layers,
                inject_single_layers=config.anchor_inject_single_layers,
                anchor_downsample=config.anchor_downsample,
                anchor_tokens=config.anchor_tokens,
                gate_init=getattr(config, "anchor_gate_init", -6.0),
            )

        self.initialize_weights()

        if self.config.grad_ckpt_settings:
            self.forward = self.forward_selective_ckpt
        else:
            self.forward = self.forward_ckpt
        self._input_requires_grad = False
            

    def initialize_weights(self):
        nn.init.zeros_(self.cond_in.weight)
        nn.init.zeros_(self.cond_in.bias)
        if self.use_reg:
            if hasattr(self, 'depth_cls_in'):
                nn.init.zeros_(self.depth_cls_in.weight)
                nn.init.zeros_(self.depth_cls_in.bias)
            if hasattr(self, 'seg_cls_in'):
                nn.init.zeros_(self.seg_cls_in.weight)
                nn.init.zeros_(self.seg_cls_in.bias)
            # Zero-init the last layer of reg_proj so alignment loss starts
            # from zero output, preventing random gradient disruption to
            # pretrained weights.  The first linear layer keeps Kaiming init
            # so that gradients can flow through immediately.
            if hasattr(self, 'reg_proj'):
                nn.init.zeros_(self.reg_proj[-1].weight)
                nn.init.zeros_(self.reg_proj[-1].bias)

    def prepare_block_inputs(
        self,
        img: Tensor,
        img_ids: Tensor,
        raymap: Tensor,
        raymap_ids: Tensor,
        timesteps: Tensor,
        cond: Tensor = None,
        depth_cls: Tensor = None,
        seg_cls: Tensor = None,
    ):
        img = self.img_in(img)
        img = img + self.cond_in(cond)
        
        if self.use_multi_level_noise:
            if timesteps.dim() == 1:
                vec = self.time_in(timestep_embedding(timesteps, 256))
                vec = vec.unsqueeze(1)
            else:
                timesteps_b, timesteps_t = timesteps.shape
                timesteps_rearrange = rearrange(
                    timesteps, 'b t -> (b t)'
                )
                vec = self.time_in(timestep_embedding(timesteps_rearrange, 256))
                vec = rearrange(vec, '(b t) c -> b t c', b=timesteps_b, t=timesteps_t)
        else:
            vec = self.time_in(timestep_embedding(timesteps, 256))

        raymap = pack(raymap, self.patch_size)
        raymap = self.traj_cond_in(raymap)

        ids = torch.cat((raymap_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        if self._input_requires_grad:
            img.requires_grad_()
            raymap.requires_grad_()

        self._reg_context = None
        if self.use_reg:
            cls_embed_list = []
            if depth_cls is not None and hasattr(self, 'depth_cls_in'):
                cls_embed_list.append(self.depth_cls_in(depth_cls))
            if seg_cls is not None and hasattr(self, 'seg_cls_in'):
                cls_embed_list.append(self.seg_cls_in(seg_cls))
            if cls_embed_list:
                cls_embed = torch.cat(cls_embed_list, dim=1)  # [B, T_cls*num_modalities, hidden_size]
                num_modalities = len(cls_embed_list)
                T_cls = cls_embed.shape[1] // num_modalities

                # Infer latent frame count from img_ids
                img_ids_t = img_ids[:, :, 0]  # [B, T*H*W]
                T_latent = int(img_ids_t.max().item()) + 1

                # Reshape cls_embed: [B, num_modalities, T_cls, hidden_size]
                cls_embed = rearrange(cls_embed, 'b (m t) c -> b m t c', m=num_modalities, t=T_cls)

                # Subsample T_cls to T_latent (uniform sampling)
                if T_cls >= T_latent:
                    indices = torch.linspace(0, T_cls - 1, T_latent, device=cls_embed.device, dtype=torch.long)
                    cls_embed = cls_embed[:, :, indices, :]  # [B, num_modalities, T_latent, hidden_size]
                else:
                    pad = T_latent - T_cls
                    cls_embed = torch.cat([cls_embed, cls_embed[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)

                # Average across modalities, keep frame dimension
                cls_embed = cls_embed.mean(dim=1)  # [B, T_latent, hidden_size]

                if self.config.reg_frame_injection:
                    # Frame-aligned injection: gather per-frame CLS embed for each img token
                    frame_indices = img_ids_t.long()  # [B, T*H*W]
                    cls_embed_per_token = cls_embed.gather(
                        1, frame_indices.unsqueeze(-1).expand(-1, -1, cls_embed.shape[-1])
                    )  # [B, T*H*W, hidden_size]
                    img = img + cls_embed_per_token
                else:
                    # Category-aligned injection: global mean pool → broadcast to all tokens
                    cls_embed_global = cls_embed.mean(dim=1, keepdim=True)  # [B, 1, hidden_size]
                    cls_embed_broadcast = cls_embed_global.expand(-1, img.shape[1], -1)  # [B, T*H*W, hidden_size]
                    img = img + cls_embed_broadcast

        return img, raymap, vec, pe

    def enable_input_require_grads(self):
        self._input_requires_grad = True

    def prepare_align_ref(self, cls_ref_list, img_ids):
        """Build the frame-aligned REG reference from per-modality DINOv3 CLS features.

        Mirrors the injection path (prepare_block_inputs): each entry of
        cls_ref_list is [B, T_cls, D_ref] (one modality, per PIXEL-frame
        references). This stacks modalities, subsamples T_cls down to the
        latent frame count inferred from img_ids (same linspace convention as
        the injection path), then averages modalities, returning
        [B, T_latent, D_ref].

        Why this is needed: img_hidden in compute_alignment_loss is
        [B(=views), T_latent*H*W, D]. The old code concatenated refs along
        dim=1 ([B, m*T_cls, D_ref]) and inferred num_cls from that, which only
        reshapes without error when m*T_cls divides T_latent*H*W (T=1 runs);
        at T=13 (T_cls=13, m=2 -> 26) it crashes on 1400 tokens, and even at
        T=1 the pooling grouped token halves by modality instead of by frame.
        With this helper, num_cls == T_latent and pooling is exactly per-frame.
        """
        img_ids_t = img_ids[:, :, 0]  # [B, T*H*W] latent frame indices
        T_latent = int(img_ids_t.max().item()) + 1
        refs = torch.stack(cls_ref_list, dim=1)  # [B, m, T_cls, D_ref]
        T_cls = refs.shape[2]
        if T_cls >= T_latent:
            indices = torch.linspace(0, T_cls - 1, T_latent, device=refs.device, dtype=torch.long)
            refs = refs[:, :, indices, :]
        else:
            pad = T_latent - T_cls
            refs = torch.cat([refs, refs[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
        return refs.mean(dim=1)  # [B, T_latent, D_ref]

    def compute_alignment_loss(self, img_hidden: Tensor, cls_ref: Tensor) -> Tensor:
        """
        Compute alignment loss between model hidden states and DINO CLS references.

        Two modes:
        1. Original REG (reg_frame_aligned=False): Global mean pool all tokens → single vector → cos_sim.
           Loses frame/view structure.
        2. Frame-aligned REG (reg_frame_aligned=True): Mean pool per-frame per-view → cos_sim per frame → average.
           Preserves temporal-spatial information for better video consistency.

        Args:
            img_hidden: [B, T_latent*H*W, D] - model hidden states at reg_align_layer
                        (B holds views; tokens are latent frames x spatial)
            cls_ref: [B, T_latent, D_ref] - per-frame DINO reference (built by
                     prepare_align_ref so num_cls == T_latent and the reshape below is exact)
        """
        if self.config.reg_frame_aligned:
            # Frame-aligned alignment loss
            # img_hidden: [B, T*V*H*W, D] → reshape to [B, T*V, H*W, D] → mean pool over H*W
            B, N, D = img_hidden.shape
            num_cls = cls_ref.shape[1]  # T*V
            num_spatial = N // num_cls  # H*W

            img_reshaped = img_hidden.reshape(B, num_cls, num_spatial, D)
            img_pooled = img_reshaped.mean(dim=2)  # [B, T*V, D]

            # Project and normalize
            img_proj = self.reg_proj(img_pooled)  # [B, T*V, D_proj]
            img_proj = F.normalize(img_proj.float(), dim=-1, eps=1e-6)
            cls_ref_normed = F.normalize(cls_ref.float(), dim=-1, eps=1e-6)

            # Per-frame cos_sim, then average
            cos_sim = torch.sum(img_proj * cls_ref_normed.detach(), dim=-1)  # [B, T*V]
            loss = 2 - 2 * torch.mean(cos_sim)
        else:
            # Original REG: global mean pool
            img_pooled = img_hidden.mean(dim=1)
            img_proj = self.reg_proj(img_pooled)
            # eps prevents NaN when reg_proj outputs zeros (early training after zero-init)
            img_proj = F.normalize(img_proj.float(), dim=-1, eps=1e-6)
            cls_ref_pooled = F.normalize(cls_ref.mean(dim=1).float(), dim=-1)
            cos_sim = torch.sum(img_proj * cls_ref_pooled.detach(), dim=-1)
            loss = 2 - 2 * torch.mean(cos_sim)

        return loss

    def forward_ckpt(
        self,
        img: Tensor,
        img_ids: Tensor,
        raymap: Tensor,
        raymap_ids: Tensor,
        timesteps: Tensor,
        cond: Tensor = None,
        depth_cls: Tensor = None,
        seg_cls: Tensor = None,
        depth_cls_ref: Tensor = None,
        seg_cls_ref: Tensor = None,
        **kwargs,
    ) -> Tensor:
        img, raymap, vec, pe = self.prepare_block_inputs(
            img, img_ids, raymap, raymap_ids, timesteps, cond, depth_cls, seg_cls
        )
        has_grad = img.grad_fn is not None

        align_loss = None

        # Reset anchor cache at the start of each forward pass
        if self.use_anchor_cache:
            self.anchor_cache.reset_cache()

        img_ids_t = img_ids[:, :, 0]  # [B, T*H*W] frame indices

        for i, block in enumerate(self.double_blocks):
            if has_grad:
                img, raymap = auto_grad_checkpoint(block, img, raymap, vec, pe)
            else:
                img, raymap = block(img, raymap, vec, pe)
            
            # Anchor KV Cache: extract at anchor layer, inject at designated layers
            if self.use_anchor_cache:
                if self.anchor_cache.should_extract(i, is_double=True):
                    for frame_id in self.anchor_cache.history_frames:
                        self.anchor_cache.extract_and_store(img, img_ids_t, i, frame_id=frame_id, is_double=True)
                if self.anchor_cache.should_inject(i, is_double=True):
                    img = self.anchor_cache.inject(img, i, is_double=True)

            if self.use_reg and i == self.reg_align_layer:
                cls_ref_list = []
                if depth_cls_ref is not None:
                    cls_ref_list.append(depth_cls_ref)
                if seg_cls_ref is not None:
                    cls_ref_list.append(seg_cls_ref)
                if cls_ref_list and self.reg_use_align_loss:
                    # Frame-aligned reference: [B, m, T_cls, D_ref] -> [B, T_latent, D_ref]
                    # so compute_alignment_loss pools exactly per latent frame
                    cls_ref_cat = self.prepare_align_ref(cls_ref_list, img_ids)
                    align_loss = self.compute_alignment_loss(img, cls_ref_cat)

        img = torch.cat((raymap, img), 1)
        raymap_len = raymap.shape[1]
        for i, block in enumerate(self.single_blocks):
            if has_grad:
                img = auto_grad_checkpoint(block, img, vec, pe)
            else:
                img = block(img, vec, pe)
            
            # Anchor KV Cache injection in single blocks (only img tokens, skip raymap)
            if self.use_anchor_cache and self.anchor_cache.should_inject(i, is_double=False):
                img = self.anchor_cache.inject(img, i, is_double=False, raymap_len=raymap_len)

        img = img[:, raymap_len:, ...]

        img = self.final_layer(img, vec)

        if self.use_reg and align_loss is not None:
            return img, align_loss
        return img

    def forward_selective_ckpt(
        self,
        img: Tensor,
        img_ids: Tensor,
        raymap: Tensor,
        raymap_ids: Tensor,
        timesteps: Tensor,
        cond: Tensor = None,
        depth_cls: Tensor = None,
        seg_cls: Tensor = None,
        depth_cls_ref: Tensor = None,
        seg_cls_ref: Tensor = None,
        **kwargs,
    ) -> Tensor:

        img, raymap, vec, pe = self.prepare_block_inputs(
            img, img_ids, raymap, raymap_ids, timesteps, cond, depth_cls, seg_cls
        )
        has_grad = img.grad_fn is not None

        align_loss = None

        # Reset anchor cache at the start of each forward pass
        if self.use_anchor_cache:
            self.anchor_cache.reset_cache()

        img_ids_t = img_ids[:, :, 0]  # [B, T*H*W] frame indices

        ckpt_depth_double = self.config.grad_ckpt_settings[0]
        for i, block in enumerate(self.double_blocks[:ckpt_depth_double]):
            if has_grad:
                img, raymap = auto_grad_checkpoint(block, img, raymap, vec, pe)
            else:
                img, raymap = block(img, raymap, vec, pe)
            
            # Anchor KV Cache
            if self.use_anchor_cache:
                if self.anchor_cache.should_extract(i, is_double=True):
                    self.anchor_cache.extract_and_store(img, img_ids_t, i, is_double=True)
                if self.anchor_cache.should_inject(i, is_double=True):
                    img = self.anchor_cache.inject(img, i, is_double=True)
            
            if self.use_reg and i == self.reg_align_layer:
                cls_ref_list = []
                if depth_cls_ref is not None:
                    cls_ref_list.append(depth_cls_ref)
                if seg_cls_ref is not None:
                    cls_ref_list.append(seg_cls_ref)
                if cls_ref_list and self.reg_use_align_loss:
                    # Frame-aligned reference: [B, m, T_cls, D_ref] -> [B, T_latent, D_ref]
                    # so compute_alignment_loss pools exactly per latent frame
                    cls_ref_cat = self.prepare_align_ref(cls_ref_list, img_ids)
                    align_loss = self.compute_alignment_loss(img, cls_ref_cat)

        for i, block in enumerate(self.double_blocks[ckpt_depth_double:], start=ckpt_depth_double):
            img, raymap = block(img, raymap, vec, pe)
            
            # Anchor KV Cache
            if self.use_anchor_cache:
                if self.anchor_cache.should_extract(i, is_double=True):
                    self.anchor_cache.extract_and_store(img, img_ids_t, i, is_double=True)
                if self.anchor_cache.should_inject(i, is_double=True):
                    img = self.anchor_cache.inject(img, i, is_double=True)
            
            if self.use_reg and i == self.reg_align_layer:
                cls_ref_list = []
                if depth_cls_ref is not None:
                    cls_ref_list.append(depth_cls_ref)
                if seg_cls_ref is not None:
                    cls_ref_list.append(seg_cls_ref)
                if cls_ref_list and self.reg_use_align_loss:
                    # Frame-aligned reference: [B, m, T_cls, D_ref] -> [B, T_latent, D_ref]
                    # so compute_alignment_loss pools exactly per latent frame
                    cls_ref_cat = self.prepare_align_ref(cls_ref_list, img_ids)
                    align_loss = self.compute_alignment_loss(img, cls_ref_cat)

        ckpt_depth_single = self.config.grad_ckpt_settings[1]

        img = torch.cat((raymap, img), 1)
        raymap_len = raymap.shape[1]
        for i, block in enumerate(self.single_blocks[:ckpt_depth_single]):
            if has_grad:
                img = auto_grad_checkpoint(block, img, vec, pe)
            else:
                img = block(img, vec, pe)
            
            if self.use_anchor_cache and self.anchor_cache.should_inject(i, is_double=False):
                img = self.anchor_cache.inject(img, i, is_double=False, raymap_len=raymap_len)

        for i, block in enumerate(self.single_blocks[ckpt_depth_single:], start=ckpt_depth_single):
            img = block(img, vec, pe)
            
            if self.use_anchor_cache and self.anchor_cache.should_inject(i, is_double=False):
                img = self.anchor_cache.inject(img, i, is_double=False, raymap_len=raymap_len)

        img = img[:, raymap_len:, ...]

        img = self.final_layer(img, vec)

        if self.use_reg and align_loss is not None:
            return img, align_loss
        return img


@MODELS.register_module("flux")
def Flux(
    cache_dir: str = None,
    from_pretrained: str = None,
    device_map: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    strict_load: bool = False,
    **kwargs,
) -> MMDiTModel:
    config = MMDiTConfig(
        from_pretrained=from_pretrained,
        cache_dir=cache_dir,
        **kwargs,
    )
    low_precision_init = from_pretrained is not None and len(from_pretrained) > 0
    if low_precision_init:
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch_dtype)
    with torch.device(device_map):
        model = MMDiTModel(config)
    if low_precision_init:
        torch.set_default_dtype(default_dtype)
    else:
        model = model.to(torch_dtype)
    if from_pretrained:
        # When transitioning from stage1 (in_channels=64) to stage2 (in_channels=192),
        # strict_load=False would skip img_in entirely, losing pretrained RGB weights.
        # Instead, we load the checkpoint manually and expand img_in weights.
        ckpt_path = from_pretrained
        if not os.path.exists(ckpt_path):
            from omninwm.utils.ckpt import load_from_hf_hub
            ckpt_path = load_from_hf_hub(ckpt_path, cache_dir)

        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt_state = load_file(ckpt_path, device=torch.cuda.current_device())
        elif ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
            ckpt_state = torch.load(ckpt_path, map_location=device_map)
        else:
            # Sharded checkpoint: use standard loading
            model = load_checkpoint(
                model, from_pretrained, cache_dir=cache_dir,
                device_map=device_map, strict=strict_load,
            )
            ckpt_state = None

        if ckpt_state is not None:
            model_sd = model.state_dict()
            # Expand img_in weights: [3072, 64] -> [3072, 192]
            if 'img_in.weight' in ckpt_state and 'img_in.weight' in model_sd:
                ckpt_shape = ckpt_state['img_in.weight'].shape
                model_shape = model_sd['img_in.weight'].shape
                if ckpt_shape != model_shape and ckpt_shape[0] == model_shape[0]:
                    old_w = ckpt_state['img_in.weight']
                    new_w = torch.zeros(model_shape, dtype=old_w.dtype, device=old_w.device)
                    new_w[:, :ckpt_shape[1]] = old_w
                    ckpt_state['img_in.weight'] = new_w

            # Expand final_layer.linear weights: [256, 3072] -> [768, 3072]
            if 'final_layer.linear.weight' in ckpt_state and 'final_layer.linear.weight' in model_sd:
                ckpt_shape = ckpt_state['final_layer.linear.weight'].shape
                model_shape = model_sd['final_layer.linear.weight'].shape
                if ckpt_shape != model_shape and ckpt_shape[1] == model_shape[1]:
                    old_w = ckpt_state['final_layer.linear.weight']
                    new_w = torch.zeros(model_shape, dtype=old_w.dtype, device=old_w.device)
                    new_w[:ckpt_shape[0]] = old_w
                    ckpt_state['final_layer.linear.weight'] = new_w
                    # Expand bias too
                    if 'final_layer.linear.bias' in ckpt_state:
                        old_b = ckpt_state['final_layer.linear.bias']
                        new_b = torch.zeros(model_shape[0], dtype=old_b.dtype, device=old_b.device)
                        new_b[:ckpt_shape[0]] = old_b
                        ckpt_state['final_layer.linear.bias'] = new_b

            # Load with strict=False to allow missing/new keys (e.g., depth_cls_in, seg_cls_in)
            missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
            from omninwm.utils.ckpt import print_load_warning
            print_load_warning(missing, unexpected)
    return model
