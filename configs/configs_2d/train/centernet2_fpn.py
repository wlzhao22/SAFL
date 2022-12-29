## REAL CUT NIGHT BOTTOM
height = 384 # 480 or 384
width = 1280  # 832 or 896
mean = [98.510154, 97.019586, 90.910471]
std = [52.464124, 53.402109, 53.902908]
collect_keys=['img', "gt_bboxes_2d", "gt_labels"]
meta_keys=(
    'filename', 'ori_filename', 
    # 'ori_shape',
    'img_info',
    'img_shape', 'pad_shape', 
    'img_norm_cfg',
    'calib',
    'flip'
)
input_modality = dict(use_lidar=False, use_camera=True)

classes = ['Car', 'Van', 'Truck']
num_classes = len(classes)

model = dict(
    type='CenterNetFPN',
    pretrained=None,
    collect_keys=collect_keys,
    backbones=dict(
        feature_backbone=dict(
            type='DLASeg',
            base_name='dla34',    
            pretrained=True,  
            down_ratio=8,
            last_level=5,
            use_dcn=True, 
            return_single_layer=False
        )),
    necks=dict(
        bifpn=dict(
            type='BiFPN',
            compound_coef=1,
            fpn_num_filters=256,
            conv_channel_coef=[128, 256, 512],
            fpn_cell_repeats=3,
        )),
    headers=dict(       
        centernet_head=dict(
            type='CenterNetFPNHead',
            num_classes=num_classes,
            in_channel=256, 
            input_shape=[1,2,3,], 
            image_size=[height, width],
            strides=[8, 16, 32],
            sizes_of_interest = [[0, 80], [64, 160], [128, 320]],
            only_proposal=True,
        )),
    roi_headers=dict(
        cascade_roi_head=dict(
            type='CustomCascadeROIHeads',
            in_channels=[256, 256, 256],
            pooler_resolution=7,
            pooler_scales=(1.0/8, 1.0/16, 1.0/32),
            sampling_ratio=0,
            pooler_type='ROIAlignV2',
            cascade_bbox_reg_weights=((10.0, 10.0, 5.0, 5.0),
                                    (20.0, 20.0, 10.0, 10.0),
                                    (30.0, 30.0, 15.0, 15.0)),
            cascade_ious=(0.5, 0.6, 0.7),
            num_classes=num_classes,
            proposal_append_gt=True,
            mult_proposal_score=True,
            batch_size_per_image=512,
            positive_fraction=0.25,
        )
    ),
)
cudnn_benchmark = True

train_cfg=None
test_cfg=None

img_norm_cfg = dict(
    mean=mean, std=std, to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile',),
    dict(type='LoadAnnotations', with_bbox2d=True, with_label=True),
    dict(type='ObjectNameFilter', classes=classes),
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
    dict(type='Pad', size=(height, width)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
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
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=meta_keys),
        ])
]

dataset_type = 'KittiDataset'
data_root = 'data/kitti'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/kitti_infos_train.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=train_pipeline,
        modality=input_modality,
        classes=classes,
        test_mode=False,
        box_type_3d='camera'
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        # classes=classes,
        classes=('Car',),
        test_mode=True,
        box_type_3d='camera'
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=classes,
        test_mode=True,
        box_type_3d='camera'
    )
)

# optimizer = dict(type='AdamW', lr=0.0003, weight_decay=0.00001)
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.00001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

momentum_config = None
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 1000,
    step=[160, 180],
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

load_from = None
resume_from = None
work_dir = 'work_dirs/BDNet_3D_Monoflex/work_dirs/20210817_dla/'

workflow = [('train', 1)]

evaluation = dict(interval=9999)
find_unused_parameters = True 
