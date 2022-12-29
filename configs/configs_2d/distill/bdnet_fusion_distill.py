## REAL CUT NIGHT BOTTOM
height = 384 # 480 or 384
width = 768  # 832 or 896
mean = [98.510154, 97.019586, 90.910471]
std = [52.464124, 53.402109, 53.902908]
gpus = 3
collect_keys=['img', "gt_bboxes_2d", "gt_labels", "img_aug", "gt_free_space", "gt_lane_seg", "gt_lane_exist", "img_affine", \
               "gt_free_space_resized", "gt_reids", "gt_reids_c", "gt_bboxes_2d_c", "gt_labels_c", "img_c"]

num_classes = 6

model = dict(
    type='BDNetDistill',
    pretrained=None,
    collect_keys=collect_keys,
    train_student=True,
    teacher=dict(
        backbones=dict(
            feature_backbone=dict(
                type='RepVGG',
                num_blocks=[4, 6, 16, 1],
                width_multiplier=[1, 1, 1, 2.5],
                override_groups_map=None,
                deploy=False,
            )),
        necks=dict(
            fpn=dict(
                type='YoloNeck',   
                depth_multiple=0.33,
                width_multiple=1.0,
                out_channels=[512, 256, 256, 512],
                in_channels=[1280, 256, 128, 64],
            ),
            fpn_env=dict(
                type='YoloNeck',
                depth_multiple=0.33,
                width_multiple=1.0,
                out_channels=[512, 256, 256, 512],
                in_channels=[1280, 256, 128, 64],
            )
            ),
        headers=dict(       
            lane_head=dict(
                type='LaneHeadTea'),

            free_space_head=dict(
                type='FreeSpaceHeadTea',
                inplanes=128),

            dynamic_bbox_head=dict(
                type='YOLOv5HeadTea',
                in_channels=[512, 256, 256],
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
                type='CenterNetHeadTea',
                in_channels=256,
                start_index=num_classes),
            
            track_head=dict(
                type='TrackHeadTea',
                in_channels=256,
                num_ids=5983,
                ),
)),
    student=dict(
        backbones=dict(
            feature_backbone=dict(
                type='RepVGG',
                num_blocks=[2, 4, 14, 1],
                width_multiplier=[0.5, 0.5, 0.5, 1],
                override_groups_map=None,
                deploy=False,
            )),
        necks=dict(
            fpn=dict(
                type='YoloNeck',   
                depth_multiple=0.33,
                width_multiple=0.25,
                out_channels=[512, 256, 256, 512],
                in_channels=[512, 128, 64, 32],
            ),
            fpn_env=dict(
                type='YoloNeck',
                depth_multiple=0.33,
                width_multiple=0.25,
                out_channels=[512, 256, 256, 512],
                in_channels=[512, 128, 64, 32],
            )
            ),
        headers=dict(       
            lane_head=dict(
                type='LaneHeadStu',
                in_channels=[64,64]),

            free_space_head=dict(
                type='FreeSpaceHeadStu',
                inplanes=64),

            dynamic_bbox_head=dict(
                type='YOLOv5HeadStu',
                in_channels=[128, 64, 64],
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
                type='CenterNetHeadStu',
                in_channels=64,
                start_index=num_classes),
            
            track_head=dict(
                type='TrackHeadStu',
                in_channels=64,
                num_ids=5983,
                ),
))
)
cudnn_benchmark = True

train_cfg, test_cfg = None, None

img_norm_cfg = dict(
    mean=mean, std=std, to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', with_mixup=True, with_reid=True),
    dict(type='LoadAnnotations', with_affine=True, with_bbox2d=True, with_label=True, with_laneSeg=True, with_free_space=True, with_mixup=True, with_reid=True),
    # dict(type='MixUp'),
    dict(type='Resize', img_scale=(width, height), keep_ratio=False),
    dict(type='RandomLaneCrop', crop_size=(50, 100, 50, 200), crop_ratio=0.5, crop_num=3),
    dict(type='LaneRandomRotate', angle=15),
    dict(type='RandomLROffsetLABEL', max_offset=50),
    dict(type='RandomUDoffsetLABEL', max_offset=50),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomAffine',with_reid=True),
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
# classes = (['Sedan','Van','STruck','MTruck','LTruck','Bus'])
classes = (['Sedan', 'Van', 'STruck', 'MTruck', 'LTruck', 'Bus','Traffic_Cone','Traffic_Bar','Traffic_Barrier'])
data_root = '/home/boden/Dev/ws/BDPilotDataset/BDMerge'
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

optimizer = dict(type='Adam', lr=0.0003, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    # step=[50, 100, 150, 200, 250],
    step=[60, 80],
    gamma=0.1,
)

checkpoint_config = dict(interval=10)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# yapf:enable
# runtime settings
total_epochs = 100
dist_params = dict(backend='nccl')
log_level = 'INFO'


load_from = '/home/boden/Dev/ws/whz/BDPilot-distill/work_dirs/20210616/teacher/epoch_90.pth'
# load_from = None
# resume_from = '/home/boden/Dev/ws/whz/BDPilot-distill/work_dirs/20210616/latest.pth'
resume_from = None

work_dir = '/home/boden/Dev/ws/whz/BDPilot-distill/work_dirs/20210618'

workflow = [('train', 1)]
