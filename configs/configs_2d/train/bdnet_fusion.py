## REAL CUT NIGHT BOTTOM
height = 384 # 480 or 384
width = 768  # 832 or 896
mean = [98.510154, 97.019586, 90.910471]
std = [52.464124, 53.402109, 53.902908]
gpus = 17
collect_keys=['img', "gt_bboxes_2d", "gt_labels", "img_aug", "gt_free_space", "gt_lane_seg", "gt_lane_exist", "img_affine", "gt_free_space_resized", "gt_reids", "gt_reids_c", "gt_bboxes_2d_c", "gt_labels_c", "img_c"]

num_classes = 6

model = dict(
    type='BDNetFusion',
    pretrained=None,
    collect_keys=collect_keys,
    backbones=dict(
        feature_backbone=dict(
            type='RepVGG',
            num_blocks=[2, 4, 14, 1],
            width_multiplier=[0.75, 0.75, 0.5, 2],
            override_groups_map=None,
            deploy=False,
        )),
    necks=dict(
        bifpn=dict(
            type='YoloNeck',
            depth_multiple=0.33,
            width_multiple=0.50,
            out_channels=[512, 256, 256, 512],
            in_channels=[1024, 128, 96, 48],
        ),
        # bifpn_env=dict(
        #     type='YoloNeck',
        #     depth_multiple=0.33,
        #     width_multiple=0.50,
        #     out_channels=[512, 256, 256, 512],
        #     in_channels=[1024, 128, 96, 48],
        # )
        ),
    headers=dict(       
        # lane_head=dict(
        #     type='LaneHead',),

        # free_space_head=dict(
        #     type='FreeSpaceHead',),

        dynamic_bbox_head=dict(
            type='YOLOv5Head',
            in_channels=[256, 128, 128],
            num_class=num_classes,
            img_shape=[height, width],
            
            # Org
            anchors= [[12, 16], [19, 36], [40, 28],  # P3/8
                      [36, 75], [76, 55], [72, 146],  # P4/16
                      [142, 110], [192, 243], [459, 401]],  # P5/32,
            
            # Kitti
            # anchors = [[28,23], [42,34], [70,31],  # P3/8
            #             [64,52], [114,50], [120,83],  # P4/16
            #             [192,91], [231,152], [333,173]], # P5/32,
            
            # Ningbo
            # anchors = [[11,12], [20,21], [11,49],  
            #             [31,33],  [50,56],  [90,86],  
            #             [173,141],  [311,189],  [468,358]],
            with_ddd=False,
            ),
        
        static_bbox_head=dict(
            type='CenterNetHead',
            in_channels=128,
            start_index=num_classes),
        
        track_head=dict(
            type='TrackHead',
            in_channels=128,
            num_ids=5983,
            ),
))
cudnn_benchmark = True

train_cfg, test_cfg = None, None

img_norm_cfg = dict(
    mean=mean, std=std, to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', with_mixup=True, with_reid=True),
    dict(type='LoadAnnotations', with_reid=True, with_affine=True, with_bbox2d=True, with_label=True, with_laneSeg=True, with_free_space=True, with_mixup=True),
    # dict(type='MixUp', with_reid=True),
    dict(type='Resize', img_scale=(width, height), keep_ratio=False),
    dict(type='RandomLaneCrop', crop_size=(50, 100, 50, 200), crop_ratio=0.5, crop_num=3),
    dict(type='LaneRandomRotate', angle=15),
    dict(type='RandomLROffsetLABEL', max_offset=50),
    dict(type='RandomUDoffsetLABEL', max_offset=50),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomAffine', with_reid=True),
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
    # dict(type='Pad', size_divisor=32),
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
classes = (['Sedan', 'Van', 'STruck', 'MTruck', 'LTruck', 'Bus','Traffic_Cone','Traffic_Bar','Traffic_Barrier'])
data_root = '/home/boden/Dev/ws/BDPilotDataset/BDMerge/'
data = dict(
    samples_per_gpu=gpus,
    workers_per_gpu=2,
    train=dict(
       type=dataset_type,
        classes = classes,
        ann_file='Annotations/BDPilot_annotations_crop_640_corner_v2.json',
        data_root = data_root,
        img_prefix = 'images_crop_640',
        lane_seg_prefix='GroundTruth/lane_thickness_3_640',
        free_space_prefix = 'GroundTruth/free_space_modified_with_chetou_1_8',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes = classes,
        ann_file='Annotations/BDPilot_annotations_crop_640_corner_v2.json',
        data_root = data_root,
        img_prefix = 'images_crop_640',
        lane_seg_prefix='GroundTruth/lane_thickness_3_640',
        free_space_prefix = 'GroundTruth/free_space_modified_with_chetou_1_8',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes = classes,
        ann_file='Annotations/BDPilot_annotations_crop_640_corner_v2.json',
        data_root = data_root,
        img_prefix = 'images_crop_640',
        lane_seg_prefix='GroundTruth/lane_thickness_3_640',
        free_space_prefix = 'GroundTruth/free_space_modified_with_chetou_1_8',
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

# load_from = '/home/boden/Dev/ws/ws/BDNet_solo_lane/work_dirs/20210531/epoch_161.pth'
# load_from = '/home/boden/Dev/ws/ws/BDNet_solo_lane/work_dirs/20210601/epoch_158.pth'
load_from = '/home/boden/Dev/ws/ws/BDNet_tracking/work_dirs/20210610/epoch_149.pth'
# load_from = None
resume_from = None

work_dir = '/home/boden/Dev/ws/ws/BDNet_tracking/work_dirs/20210611'

workflow = [('train', 1)]
