# Mask2Former instance segmentation on the Roboflow "hockey-rink-landmarks" v2
# dataset. Mirrors configs/mask2former_ice_rink/, but segments the 11 painted
# rink features instead of the single playing surface.
_base_ = ["../mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py"]

num_things_classes = 11
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

# max_per_image is for instance segmentation. The densest image in the dataset
# carries 18 landmarks, so 50 leaves plenty of headroom.
max_per_image = 50

image_size = (1024, 1024)
batch_augments = [
    dict(
        type="BatchFixedSizePad",
        size=image_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False,
    )
]
data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    batch_augments=batch_augments,
)
model = dict(
    data_preprocessor=data_preprocessor,
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1]),
    ),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes, num_stuff_classes=num_stuff_classes
    ),
    test_cfg=dict(
        panoptic_on=False,
        max_per_image=max_per_image,
    ),
)

# dataset settings
train_pipeline = [
    dict(
        type="LoadImageFromFile", to_float32=True, backend_args={{_base_.backend_args}}
    ),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="RandomFlip", prob=0.5),
    # large scale jittering
    dict(
        type="RandomResize",
        scale=image_size,
        ratio_range=(0.1, 2.0),
        resize_type="Resize",
        keep_ratio=True,
    ),
    dict(
        type="RandomCrop",
        crop_size=image_size,
        crop_type="absolute",
        recompute_bbox=True,
        allow_negative_crop=True,
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type="PackDetInputs"),
]

# Roboflow already stretched every image to 1008x1008, so evaluate at native
# resolution. Downscaling to the COCO default of 800px costs recall on the
# thinnest classes (Faceoff Dot, Goal Line).
test_pipeline = [
    dict(
        type="LoadImageFromFile", to_float32=True, backend_args={{_base_.backend_args}}
    ),
    dict(type="Resize", scale=(1008, 1008), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

dataset_type = "CocoRinkLandmarksDataset"
data_root = "data/HockeyRinkLandmarks/"
batch_size = 2
train_dataloader = dict(
    sampler=dict(type="InfiniteSampler"),
    batch_size=batch_size,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="train/_annotations.coco.json",
        # _delete_ drops the panoptic base's `seg` prefix, which would otherwise
        # survive the dict merge and point at a COCO path that does not exist.
        data_prefix=dict(_delete_=True, img="train/"),
        pipeline=train_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="test/_annotations.coco.json",
        data_prefix=dict(_delete_=True, img="test/"),
        pipeline=test_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="valid/_annotations.coco.json",
        data_prefix=dict(_delete_=True, img="valid/"),
        pipeline=test_pipeline,
    ),
)

# `classwise` matters here: Slot Box and Trapezoid have ~30 training instances
# each, so the headline mAP hides how they are really doing.
val_evaluator = dict(
    _delete_=True,
    type="CocoMetric",
    ann_file=data_root + "valid/_annotations.coco.json",
    metric=["bbox", "segm"],
    classwise=True,
    format_only=False,
    backend_args={{_base_.backend_args}},
)
test_evaluator = dict(
    _delete_=True,
    type="CocoMetric",
    ann_file=data_root + "test/_annotations.coco.json",
    metric=["bbox", "segm"],
    classwise=True,
    format_only=False,
    backend_args={{_base_.backend_args}},
)

max_iters = 120000

# The inherited schedule decays at 327778/355092 -- milestones for the 368750
# iteration COCO recipe, which never fire in a 120k run. Rescale them to the
# same 89%/96% points so the run actually gets its LR drops.
param_scheduler = dict(
    type="MultiStepLR",
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[106667, 115556],
    gamma=0.1,
)

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=max_iters,
    val_interval=1500,
    dynamic_intervals=None,
)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=5,
        save_best="coco/segm_mAP",
        rule="greater",
    )
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)
