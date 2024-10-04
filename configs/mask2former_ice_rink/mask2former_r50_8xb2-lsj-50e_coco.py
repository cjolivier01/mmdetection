# _base_ = ['./mask2former_r50_8xb2-lsj-50e_coco-panoptic.py']
_base_ = ["../mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py"]

num_things_classes = 80  # Just an ice rink
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

# max_per_image is for instance segmentation.
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

test_pipeline = [
    dict(
        type="LoadImageFromFile", to_float32=True, backend_args={{_base_.backend_args}}
    ),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

dataset_type = "CocoIceRink2Dataset"
data_root = "data/IceRink/"

train_dataloader = dict(
    sampler=dict(type="InfiniteSampler"),
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="train/_annotations.coco.json",
        data_prefix=dict(img="train/"),
        pipeline=train_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="test/_annotations.coco.json",
        data_prefix=dict(img="test/"),
        pipeline=test_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="valid/_annotations.coco.json",
        data_prefix=dict(img="valid/"),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    _delete_=True,
    type="CocoMetric",
    ann_file=data_root + "valid/_annotations.coco.json",
    metric=["bbox", "segm"],
    format_only=False,
    backend_args={{_base_.backend_args}},
)
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=2000, max_keep_ckpts=25)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=30000,
    val_interval=1500,
    dynamic_intervals=None,
)
