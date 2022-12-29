
dataset_type = 'KittiDataset'
input_modality = dict(use_lidar=True, use_camera=True)


height = 384
width = 1280
mean = [98.510154, 97.019586, 90.910471]
std = [52.464124, 53.402109, 53.902908]
collect_keys=['img', "gt_bboxes_2d", "gt_labels", 
    "img_aug", 
]
meta_keys=(
    'filename', 'ori_filename', 
    'ori_shape',
    'img_info',
    'img_shape', 'pad_shape', 
    #'scale_factor', 'flip', 'flip_direction', 
    'img_norm_cfg'
)

num_classes = 2
class_names = ['car', ]

train_cfg = None
test_cfg = dict(
    topk=100, local_maximum_kernel=3,
    max_per_img=200,
    nms_cfg=dict(
        class_agnostic=False, 
        max_num=200, 
        iou_threshold=0.5,
        score_threshold=0.15
    ),
)

model = dict(
    type='CMDMono',
    pretrained=None,
    collect_keys=collect_keys,
    backbones=dict(
        feature_backbone=dict(
            type='RepVGG',
            num_blocks=[2, 4, 14, 1],               
            # width_multiplier=[0.75, 0.75, 0.5, 2],     # RepVGG-?
            # width_multiplier=[0.75, 0.75, 0.75, 2.5],  # RepVGG-A0 
            width_multiplier=[1, 1, 1, 2.5],             # RepVGG-B0
            override_groups_map=None,
            deploy=False,
        )),
    necks=dict(
        bifpn=dict(
            type='YoloNeck',
            depth_multiple=0.33,
            width_multiple=0.50,
            out_channels=[256, 256, 256, 256],
            # in_channels=[1024, 128, 96, 48],            # RepVGG-? 
            in_channels=[256, 128, 64],             # RepVGG-B0
        ),
    ),
    headers=dict(    
        dynamic_bbox_head=dict(  
            type='CenterNetFPNHead',
            num_classes=num_classes,
            in_channel=128, 
            input_shape=[1,2,3,4,5], 
            image_size=[height, width],
            strides=[4, 8, 16],
            sizes_of_interest = [[0, 160], [128, 640], [512, 10000000]],
            only_proposal=False,
        )
    )
)
cudnn_benchmark = True


img_norm_cfg = dict(
    mean=mean, std=std, to_rgb=True)

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox2d=True,
        with_label=True,
        file_client_args=file_client_args
    ),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Albu', transforms=[
            dict(
                type='OneOf',
                transforms=[dict(type='RandomBrightnessContrast', p=1.0),
                            dict(type='RandomGamma', p=1.0),
                            dict(type='IAASharpen', p=1.0),
                            dict(type='JpegCompression', p=1.0),
                            dict(type='ChannelShuffle', p=1.0),
                            dict(type='HueSaturationValue',p=1.0)],
                p=0.2
            )
        ]
    ),
    dict(type='Pad', size=(height, width)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect3D',
        keys=collect_keys,
        meta_keys=meta_keys
    ),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(width, height),
        flip=False,
        transforms=[
            dict(type='Pad', size=(height, width)),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=meta_keys),
        ])
]

data_root = 'data/kitti'

db_sampler = None
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + '/kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR'
        )
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR')
)

# optimizer = dict(type='Adam', lr=0.003, betas=(0.9, 0.999), eps=1e-08)
# optimizer = dict(type='SGD', lr=1e-3)
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    # step=[50, 100, 150, 200, 250],
    step=[180, 190],
    gamma=0.1,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)

# total_epochs = 
total_epochs = 200
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

checkpoint_config = dict(interval=10)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)

# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = None
resume_from = None

work_dir = "work_dirs/kitti/CMD_mono3D_KITTI_Car/"

workflow = [('train', 1)]
evaluation = dict(interval=1)
find_unused_parameters = True
