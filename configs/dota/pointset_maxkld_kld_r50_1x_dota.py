norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='RRepPointsDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
    ),
    neck=
        dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5,
        norm_cfg=norm_cfg
        ),
    bbox_head=dict(
        type='GRepRepPointsHead',
        num_classes=16,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.3,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=2,
        norm_cfg=norm_cfg,
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox_init=dict(type='KLDLoss', loss_weight=0.375),
        loss_bbox_refine=dict(type='KLDLoss', loss_weight=1.0),
        transform_method='rotrect',
        show_points=True,
        topk=6,
        anti_factor=0.75))
# training and testing settings
train_cfg = dict(
    init=dict(
        assigner=dict(type='ConvexAssigner', scale=4, pos_num=1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),

    refine=dict(
        assigner=dict(
            type='MaxKLDAssigner',
            pos_iou_thr=0.4,
            neg_iou_thr=0.3,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))

test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='rnms', iou_thr=0.4),
    max_per_img=2000)
# dataset settings
dataset_type = 'DotaDataset'
data_root = 'data/cfa_dota_obb_1024/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CorrectBox', correct_rbbox=True, refine_rbbox=True),
    dict(type='RotateResize',
        img_scale=[(1333, 768), (1333, 1280)],
        keep_ratio=True,
        multiscale_mode='range',
        clamp_rbbox=False),
    dict(type='RotateRandomFlip', flip_ratio=0.5, direction=['horizontal']),
    dict(type='AddParams', with_xyxy=False, with_xywht=False, remove_small=False, just_remove_small=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 1024),
        flip=False,
        transforms=[
            dict(type='RotateResize', keep_ratio=True),
            dict(type='RotateRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval_split/trainval_coco.json',
        img_prefix=data_root + 'trainval_split/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval_split/trainval_coco.json',
        img_prefix=data_root + 'trainval_split/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_split/test_coco.json',
        img_prefix=data_root + 'test_split/images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

'''
overlaps = torch.exp(1 / (2 + kld.sqrt())) - 1
'''