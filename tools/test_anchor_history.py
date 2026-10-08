"""Smoke test for anchor_cache history_frames extension.
验证：
1. history_frames=[0] → 与之前行为一致
2. history_frames=[0,1,2] → cache 包含3帧拼接
3. inject 逻辑正确工作
"""

import torch
import sys
sys.path.insert(0, '/data1/qjyuan/OmniNWM')
sys.path.insert(0, '/data1/qjyuan/OmniNWM/omninwm/models/ops/occ_pooling')

from omninwm.models.mmdit.anchor_cache import AnchorKVCache


def test_v1_single_frame():
    """与 P7 行为一致"""
    cache = AnchorKVCache(
        hidden_size=128, num_heads=4, anchor_layer=0,
        inject_double_layers=[2], inject_single_layers=[3],
        gate_init=0.0, history_frames=[0],
    )
    B, T, H, W = 2, 13, 16, 24
    img_hidden = torch.randn(B, T * H * W, 128)
    img_ids_t = torch.arange(T * H * W).unsqueeze(0).expand(B, -1)
    img_ids_t = img_ids_t // (H * W)  # frame index per token

    # Only extract frame 0
    cache.extract_and_store(img_hidden, img_ids_t, layer_idx=0, frame_id=0)
    assert 0 in cache._double_cache, "frame 0 not extracted"
    assert len(cache._double_cache[0]) == 1, f"expected 1 frame, got {len(cache._double_cache[0])}"
    K, V = cache._concat_cache(cache._double_cache[0])
    assert K.shape == (B, 4, 128, 32), f"K shape {K.shape}"
    print(f"  [OK] single frame: cache K shape {K.shape}")

    # Inject
    cache.double_gates['layer_2'] = torch.nn.Parameter(torch.tensor([0.0]))
    out = cache.inject(img_hidden, layer_idx=2, is_double=True)
    assert out.shape == img_hidden.shape
    print(f"  [OK] inject preserves shape")

    # Verify inject actually modified the hidden (gate > 0)
    assert not torch.allclose(out, img_hidden)
    print(f"  [OK] inject changed hidden (gate>0)")


def test_v3_historical():
    """3 frames concatenated: cache 3x larger"""
    cache = AnchorKVCache(
        hidden_size=128, num_heads=4, anchor_layer=0,
        inject_double_layers=[2], inject_single_layers=[3],
        gate_init=0.0, history_frames=[0, 1, 2],
    )
    B, T, H, W = 2, 13, 16, 24
    img_hidden = torch.randn(B, T * H * W, 128)
    img_ids_t = (torch.arange(T * H * W).unsqueeze(0).expand(B, -1)) // (H * W)

    for f in [0, 1, 2]:
        cache.extract_and_store(img_hidden, img_ids_t, layer_idx=0, frame_id=f)
    assert len(cache._double_cache[0]) == 3, f"expected 3 frames, got {len(cache._double_cache[0])}"
    K, V = cache._concat_cache(cache._double_cache[0])
    assert K.shape == (B, 4, 3 * 128, 32), f"K shape {K.shape}, expected 3x larger"
    print(f"  [OK] 3 frames: cache K shape {K.shape}")

    cache.double_gates['layer_2'] = torch.nn.Parameter(torch.tensor([1.0]))
    out = cache.inject(img_hidden, layer_idx=2, is_double=True)
    assert out.shape == img_hidden.shape
    print(f"  [OK] 3-frame inject preserves shape")


if __name__ == '__main__':
    print("=== Test 1: single-frame history [0] (P7 compat) ===")
    test_v1_single_frame()
    print("=== Test 2: historical [0,1,2] (INSPATIO-style) ===")
    test_v3_historical()
    print("\nAll smoke tests passed!")