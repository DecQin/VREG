
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List


@dataclass
class CacheConfig:
    enable_cache: bool = False
    cache_layers: List[int] = field(default_factory=lambda: [0, 2, 4, 6, 8])
    cache_window_size: int = 4
    spatial_cache_stride: int = 1
    cache_type: str = "mean"


class SpatioTemporalCache(nn.Module):
    """时空特征缓存模块"""
    
    def __init__(self, config: CacheConfig, hidden_size: int, num_heads: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.feature_cache = {}
        self.fusion_weights = None
        if config.cache_type == "weighted_sum":
            self.fusion_weights = nn.ParameterDict({
                f"layer_{layer_idx}": nn.Parameter(torch.ones(config.cache_window_size) / config.cache_window_size)
                for layer_idx in config.cache_layers
            })
        self.reset_cache()
    
    def reset_cache(self):
        """清空缓存"""
        self.feature_cache = {}
        for layer_idx in self.config.cache_layers:
            self.feature_cache[layer_idx] = []
    
    def _apply_spatial_cache(self, features, layer_idx):
        """应用空间缓存策略"""
        if self.config.spatial_cache_stride <= 1:
            return features
        B, L, C = features.shape
        stride = self.config.spatial_cache_stride
        cached_spatial = features[:, ::stride, :].clone()
        features[:, ::stride, :] = cached_spatial
        return features
    
    def _apply_temporal_cache(self, features, layer_idx):
        """应用时间缓存策略"""
        if layer_idx not in self.feature_cache:
            self.feature_cache[layer_idx] = []
        cache_list = self.feature_cache[layer_idx]
        cache_list.append(features.clone().detach())
        if len(cache_list) > self.config.cache_window_size:
            cache_list.pop(0)
        self.feature_cache[layer_idx] = cache_list
        if not self.training or len(cache_list) < 2:
            return features, features
        if self.config.cache_type == "mean":
            cached_mean = torch.stack(cache_list[:-1], dim=0).mean(dim=0)
            fused_features = 0.5 * features + 0.5 * cached_mean
        elif self.config.cache_type == "weighted_sum" and self.fusion_weights is not None:
            weights = self.fusion_weights[f"layer_{layer_idx}"][:len(cache_list)-1]
            weights = weights / weights.sum()
            cached_weighted = torch.sum(torch.stack(cache_list[:-1], dim=0) * weights[:, None, None], dim=0)
            fused_features = features + cached_weighted
        else:
            fused_features = features
        return fused_features, cache_list[-1]
    
    def forward(self, features, layer_idx):
        """前向传播"""
        if not self.config.enable_cache or layer_idx not in self.config.cache_layers:
            return features, None
        features = self._apply_spatial_cache(features, layer_idx)
        fused_features, cache_output = self._apply_temporal_cache(features, layer_idx)
        return fused_features, cache_output


def build_cache_config(cfg_dict):
    return CacheConfig(
        enable_cache=cfg_dict.get("enable_cache", False),
        cache_layers=cfg_dict.get("cache_layers", [0, 2, 4, 6, 8]),
        cache_window_size=cfg_dict.get("cache_window_size", 4),
        spatial_cache_stride=cfg_dict.get("spatial_cache_stride", 1),
        cache_type=cfg_dict.get("cache_type", "mean")
    )

