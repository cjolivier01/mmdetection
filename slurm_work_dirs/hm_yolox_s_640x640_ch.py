num_classes = 80
detector_standalone_model = dict(
    type='YOLOX',
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=1, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
    input_size=(500, 1333),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        '/home/colivier/Downloads/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
    ))
model = dict(
    detector=dict(
        type='YOLOX',
        random_size_range=(15, 25),
        random_size_interval=10,
        backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
        neck=dict(
            type='YOLOXPAFPN',
            in_channels=[128, 256, 512],
            out_channels=128,
            num_csp_blocks=1),
        bbox_head=dict(
            type='YOLOXHead',
            num_classes=80,
            in_channels=128,
            feat_channels=128),
        train_cfg=dict(
            assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(
            score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
        input_size=(640, 640),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/home/colivier/Downloads/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
        )))
img_scale = (640, 640)
gpu_count = 8
samples_per_gpu_a100_40gb_yolox_s = 38
samples_per_gpu_a16_16gb_yolox_s = 5
samples_per_gpu_a100_40gb_yolox_l = 28
samples_per_gpu = 5
yolox_img_scale = (500, 1333)
image_pad_value = 114.0
train_pipeline = [
    dict(
        type='Mosaic',
        img_scale=(500, 1333),
        pad_val=114.0,
        bbox_clip_border=False),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-250, -667),
        bbox_clip_border=False),
    dict(
        type='MixUp',
        img_scale=(500, 1333),
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=False),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        img_scale=(500, 1333),
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(500, 1333),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_val=dict(img=(114.0, 114.0, 114.0)),
                size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(500, 1333),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_val=dict(img=(114.0, 114.0, 114.0)),
                size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
inference_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='HmCrop', keys=['img'], save_clipped_images=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(500, 1333),
        flip=False,
        transforms=[
            dict(type='HmImageToTensor', keys=['img']),
            dict(type='HmResize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='HmPad', pad_val=114.0, size_divisor=32),
            dict(type='VideoCollect', keys=['img', 'clipped_image'])
        ])
]
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file=[
                'data/crowdhuman/annotations/crowdhuman_train.json',
                'data/crowdhuman/annotations/crowdhuman_val.json'
            ],
            img_prefix=['data/crowdhuman/train', 'data/crowdhuman/val'],
            classes=('pedestrian', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False),
        pipeline=[
            dict(
                type='Mosaic',
                img_scale=(500, 1333),
                pad_val=114.0,
                bbox_clip_border=False),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-250, -667),
                bbox_clip_border=False),
            dict(
                type='MixUp',
                img_scale=(500, 1333),
                ratio_range=(0.8, 1.6),
                pad_val=114.0,
                bbox_clip_border=False),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Resize',
                img_scale=(500, 1333),
                keep_ratio=True,
                bbox_clip_border=False),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file='data/crowdhuman/annotations/val.json',
        img_prefix='data/crowdhuman/val/Images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(500, 1333),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_val=dict(img=(114.0, 114.0, 114.0)),
                        size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='data/crowdhuman/annotations/val.json',
        img_prefix='data/crowdhuman/val/Images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(500, 1333),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_val=dict(img=(114.0, 114.0, 114.0)),
                        size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    inference=dict(
        type='CocoDataset',
        ann_file='data/crowdhuman/annotations/val.json',
        img_prefix='data/crowdhuman/val/Images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='HmCrop', keys=['img'], save_clipped_images=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(500, 1333),
                flip=False,
                transforms=[
                    dict(type='HmImageToTensor', keys=['img']),
                    dict(type='HmResize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='HmPad', pad_val=114.0, size_divisor=32),
                    dict(type='VideoCollect', keys=['img', 'clipped_image'])
                ])
        ]))
log_config = dict(interval=4, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
optimizer = dict(
    type='SGD',
    lr=0.000625,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
total_epochs = 80
num_last_epochs = 10
interval = 1
runner = dict(type='EpochBasedRunner', max_epochs=80)
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,
    num_last_epochs=10,
    min_lr_ratio=0.05)
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=10, priority=48),
    dict(type='SyncNormHook', num_last_epochs=10, interval=1, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=10)
evaluation = dict(metric=['bbox'], interval=100)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
work_dir = 'slurm_work_dirs'
auto_resume = False
gpu_ids = range(0, 392)
