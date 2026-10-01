_base_ = ["./mask2former_r50_8xb2-lsj-50e_coco.py"]
pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth"  # noqa

depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        embed_dims=96,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    panoptic_head=dict(in_channels=[96, 192, 384, 768]),
    init_cfg=None,
)

# Warm-start from the COCO instance-segmentation Mask2Former (mask AP 46.1).
#
# The alternative was the ice rink checkpoint
# (work_dirs/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/iter_120000.pth),
# whose backbone has already seen 120k iterations of rink imagery. COCO wins
# here because the decoder is what transfers: 654 training images is far too
# few to teach the 100 queries multi-instance, multi-class behaviour from
# scratch, and the ice rink decoder learned the opposite prior -- emit one
# large mask per image. The domain gap in the backbone closes quickly during
# fine-tuning; a collapsed query prior does not.
#
# To try the ice rink init instead, see INIT_FROM=icerink in
# openmm/train_rink_landmarks.sh.
#
# The classification head is Linear(256, 81) in this checkpoint against
# Linear(256, 12) here; mmengine logs the size mismatch and leaves those two
# tensors randomly initialised, which is what we want. Every other weight loads.
load_from = "https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth"  # noqa

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    "backbone": dict(lr_mult=0.1, decay_mult=1.0),
    "backbone.patch_embed.norm": backbone_norm_multi,
    "backbone.norm": backbone_norm_multi,
    "absolute_pos_embed": backbone_embed_multi,
    "relative_position_bias_table": backbone_embed_multi,
    "query_embed": embed_multi,
    "query_feat": embed_multi,
    "level_embed": embed_multi,
}
custom_keys.update(
    {
        f"backbone.stages.{stage_id}.blocks.{block_id}.norm": backbone_norm_multi
        for stage_id, num_blocks in enumerate(depths)
        for block_id in range(num_blocks)
    }
)
custom_keys.update(
    {
        f"backbone.stages.{stage_id}.downsample.norm": backbone_norm_multi
        for stage_id in range(len(depths) - 1)
    }
)
# optimizer
optim_wrapper = dict(paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))
