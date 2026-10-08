# use_multi_level_noise = True
# traj_ctrl = True
# use_depth = False
# use_seg = False
# depth = 19
# depth_single_blocks = 38
# cross_view_list = list(range(depth,depth_single_blocks,9))
# mv_order_map = {
#     0: [5, 1],
#     1: [0, 2],
#     2: [1, 3],
#     3: [2, 4],
#     4: [3, 5],
#     5: [4, 0],
# }

# # Define model components
# model = dict(
#     type="flux",
#     from_pretrained = 'pretrained/Open_Sora_v2.safetensors',
#     strict_load=False,
#     fused_qkv=False,
#     use_liger_rope=True,
#     grad_ckpt_settings=(100, 100),
#     # model architecture
#     in_channels=192,
#     hidden_size=3072,
#     mlp_ratio=4.0,
#     num_heads=24,
#     depth=depth,
#     depth_single_blocks=depth_single_blocks,
#     axes_dim=[16, 56, 56],
#     theta=10_000,
#     qkv_bias=True,
#     mv_order_map=mv_order_map,
#     cross_view_list=cross_view_list,
#     use_depth = use_depth,
#     use_seg = use_seg,
#     use_multi_level_noise = use_multi_level_noise,
# )

use_multi_level_noise = True
traj_ctrl = True
use_depth = False
use_seg = False
use_reg = False
reg_align_layer = 8
reg_use_align_loss = True
dinov3_embed_dim = 768
reg_loss_weight = 1.0
reg_pred_weight = 0.1
depth = 19
depth_single_blocks = 38
cross_view_list = list(range(depth,depth_single_blocks,9))
mv_order_map = {
    0: [5, 1],
    1: [0, 2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4, 0],
}

# Define model components
model = dict(
    type="flux",
    from_pretrained = 'pretrained/Open_Sora_v2.safetensors',
    strict_load=False,
    fused_qkv=False,
    use_liger_rope=True,
    grad_ckpt_settings=(100, 100),
    # model architecture
    in_channels=192,
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
    reg_align_layer = reg_align_layer,
    reg_use_align_loss = reg_use_align_loss,
    dinov3_embed_dim = dinov3_embed_dim,
    reg_loss_weight = reg_loss_weight,
    reg_pred_weight = reg_pred_weight,
)