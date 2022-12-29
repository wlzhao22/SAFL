## REAL CUT NIGHT BOTTOM
height = 384 # 480 or 384
width = 1280  # 832 or 896
mean = [98.510154, 97.019586, 90.910471]
std = [52.464124, 53.402109, 53.902908]
gpus = 20
collect_keys=['img', "gt_bboxes_2d", "gt_labels", "img_aug",'gt_ddds_head_direction', 'gt_ddds_dx', 'gt_ddds_dw', 'gt_ddds_l0', 'gt_ddds_l1', 'gt_ddds_l2', 'gt_ddds_res_depth', 'gt_ddds_rotation', 'gt_ddds_size', 'gt_ddds_center_2d']

num_classes = 3

model = dict(
    type='BDNetFusion',
    pretrained=None,
    collect_keys=collect_keys,
    backbones=dict(
        feature_backbone=dict(
            type='RepVGG',
            num_blocks=[2, 4, 14, 1],
            width_multiplier=[0.75, 0.75, 0.75, 2.5],
            override_groups_map=None,
            deploy=False,
        )),
    necks=dict(
        bifpn=dict(
            type='YoloNeck',
            depth_multiple=0.33,
            width_multiple=0.50,
            out_channels=[512, 256, 256, 512],
            in_channels=[1280, 192, 96, 48],
        ),
        ),
    headers=dict(       
        dynamic_bbox_head=dict(
            type='YOLOv5Head',
            in_channels=[256, 128, 128],
            num_class=num_classes,
            img_shape=[height, width],
            anchors = [[28,23], [42,34], [70,31],  # P3/8
                        [64,52], [114,50], [120,83],  # P4/16
                        [192,91], [231,152], [333,173]], # P5/32,
            bbox_size_anchor = [[1.52593746, 1.62851544, 3.88421244],
                                [2.20492607, 1.90205447, 5.07570039],
                                [3.24863732, 2.58747379, 10.03602725]],
            with_ddd=True,
            ),
))
cudnn_benchmark = True

train_cfg, test_cfg = None, None

img_norm_cfg = dict(
    mean=mean, std=std, to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox2d=True, with_label=True, with_ddd=True),
    dict(type='Resize', img_scale=(width, height), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Albu', transforms=[
        dict(
            type='OneOf',
            transforms=[dict(type='RandomBrightnessContrast', p=1.0),
                        dict(type='RandomGamma', p=1.0),
                        dict(type='IAASharpen', p=1.0),
                        dict(type='JpegCompression', p=1.0),
                        dict(type='ChannelShuffle', p=1.0),
                        dict(type='HueSaturationValue',p=1.0)],p=0.3)]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
        keys=collect_keys,
        ),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(width, height),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(width, height), keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'BDDataset'
classes = (['car', 'van', 'truck'])
data_root = '/home/boden/Dev/ws/BDPilotDataset/Kitti/train/'
data = dict(
    samples_per_gpu=gpus,
    workers_per_gpu=2,
    train=dict(
       type=dataset_type,
        classes = classes,
        ann_file='annotations/annotation_3d_v3.json',
        data_root = data_root,
        img_prefix = 'images',
        with_ddd=True,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes = classes,
        ann_file='annotations/annotation_3d_v3.json',
        data_root = data_root,
        img_prefix = 'images',
        with_ddd=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes = classes,
        ann_file='annotations/annotation_3d_v3.json',
        data_root = data_root,
        img_prefix = 'images',
        with_ddd=True,
        pipeline=test_pipeline))

optimizer = dict(type='Adam', lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    # step=[50, 100, 150, 200, 250],
    step=[45, 60],
    gamma=0.1,
)

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# yapf:enable
# runtime settings
total_epochs = 200
dist_params = dict(backend='nccl')
log_level = 'INFO'

# load_from = '/home/boden/Dev/ws/ws/BDNet_3D/work_dirs/20210629/epoch_200.pth'
# load_from = '/home/boden/Dev/ws/ws/BDNet_3D/work_dirs/20210630/epoch_200.pth'
load_from = '/home/boden/Dev/ws/ws/BDNet_3D/work_dirs/20210631/epoch_7.pth'
resume_from = None

work_dir = '/home/boden/Dev/ws/ws/BDNet_3D/work_dirs/20210701'

workflow = [('train', 1)]
