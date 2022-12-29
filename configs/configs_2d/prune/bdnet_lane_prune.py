## REAL CUT NIGHT BOTTOM
height = 384 # 480 or 384
width = 768  # 832 or 896
# mean = [108.171225, 107.374997, 103.024430]
# std = [55.571922, 56.944959, 58.899119]
mean = [98.510154, 97.019586, 90.910471]
std = [52.464124, 53.402109, 53.902908]
gpus = 16
# collect_keys=['img', "gt_bboxes_2d", "gt_labels", "gt_free_space", "gt_reids", "img_aug", "gt_lane_seg", "gt_lane_exist"]
# collect_keys=['img', "img_aug", "gt_lane_seg", "gt_lane_exist", "gt_lane_change"]
collect_keys=['img', "img_aug", "gt_lane_seg", "gt_lane_exist"]


compound_coef = 1
backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
fpn_num_filters = [64, 64, 112, 160, 224, 288, 384, 384, 384]
# fpn_cell_repeats = [3, 3, 5, 6, 7, 7, 8, 8, 8]
fpn_cell_repeats = [3, 3, 5, 6, 7, 7, 8, 8, 8]
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
# pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
num_scales = len(scales)
use_p2 = True

if use_p2:
    pyramid_levels = [6, 6, 6, 6, 6, 6, 6, 6, 7]
    conv_channel_coef = {
                # the channels of P3/P4/P5.
                0: [24, 40, 112, 320],
                1: [24, 40, 112, 320],
                2: [24, 48, 120, 352],
                3: [32, 48, 136, 384],
                4: [32, 56, 160, 448],
                5: [40, 64, 176, 512],
                6: [40, 72, 200, 576],
                7: [48, 72, 200, 576],
                8: [48, 80, 224, 640], # 48?
                9: [48, 96, 128, 1024],
            }
else:
    pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
    conv_channel_coef = {
        # the channels of P2/P3/P4/P5.
        0: [40, 112, 320],
        1: [40, 112, 320],
        2: [48, 120, 352],
        3: [48, 136, 384],
        4: [56, 160, 448],
        5: [64, 176, 512],
        6: [72, 200, 576],
        7: [72, 200, 576],
        8: [80, 224, 640],
    }

# row_anchor = [320,336,352,368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624 , 640, 656, 672,688,704,720,736,752]
# row_anchor = [240, 256, 272, 288,304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624]
# row_anchor = [160,176,192,208,224,240, 256, 272, 288,304,320, 336, 352, 368, 383]
row_anchor = [192,208,224,240, 256, 272, 288,304,320, 336, 352, 368, 383]
num_anchors = len(aspect_ratios) * num_scales
num_classes = 6

model = dict(
    type='BDNetFusion',
    pretrained=None,
    collect_keys=collect_keys,
    backbones=dict(
        feature_backbone=dict(
            type='RepVGG',
            num_blocks=[2, 4, 14, 1],
            # width_multiplier=[1.5, 1.5, 1.5, 2.75],
            # width_multiplier=[1, 1, 1, 2.5],
            width_multiplier=[0.75, 0.75, 0.5, 2],
            override_groups_map=None,
            deploy=False,
        )),
    # necks=dict(
    #     bifpn=dict(
    #         type='BiFPN',
    #         compound_coef=compound_coef,
    #         fpn_num_filters=fpn_num_filters[compound_coef],
    #         conv_channel_coef=conv_channel_coef[9],
    #         fpn_cell_repeats=fpn_cell_repeats[compound_coef],
    #         onnx_export=True,
    #     )),
    headers=dict(
        # bbox_head=dict(
        #     type='DetHead',
        #     in_channels=fpn_num_filters[compound_coef],
        #     num_anchors=num_anchors,
        #     num_classes=num_classes,
        #     num_layers=1,
        #     num_ids=944,
        #     anchor_scale=anchor_scale[compound_coef],
        #     ratios=aspect_ratios,
        #     scales=scales,
        #     pyramid_levels=pyramid_levels[compound_coef],
        #     onnx_export=False,
        #     with_tracking=True),
        # free_space_head=dict(
        #     type='FreeSpaceHead',
        #     in_planes=fpn_num_filters[compound_coef]),
        # lane_head=dict(
        #     type='LaneHead',
        #     segmentation_classes=4,
        #     in_planes=fpn_num_filters[compound_coef],
        #     row_anchor=row_anchor),
        lane_head=dict(
            type='LaneHeadPrune',
            segmentation_classes=4,
            planes=96,
            row_anchor=row_anchor),
        
))
cudnn_benchmark = True

train_cfg, test_cfg = None, None

img_norm_cfg = dict(
    mean=mean, std=std, to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox2d=True, with_label=True, with_reid=True, with_free_space=True, with_laneSeg=True),
    dict(type='LoadAnnotations', with_bbox2d=False, with_label=False, with_reid=False, with_free_space=False, with_laneSeg=True),
    # dict(type='LoadAnnotations', with_car_mask=False,with_bbox2d=True,with_label=True,with_reid=True,with_free_space=True),
    dict(type='Resize', img_scale=(width, height), keep_ratio=False),
    # dict(type='RandomLaneCrop', crop_size=(50, 200, 50, 250), crop_ratio=0.5, crop_num=3),
    dict(type='RandomLaneCrop', crop_size=(50, 100, 50, 200), crop_ratio=0.5, crop_num=3),
    dict(type='LaneRandomRotate', angle=15),
    dict(type='RandomLROffsetLABEL', max_offset=100),
    dict(type='RandomUDoffsetLABEL', max_offset=100),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomCropResize',probability=0.2),
    # dict(type='RandomErasing', probability=0.3),
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
# classes = (['Car','Tram','Truck'])
# classes = (['Sedan','Van','STruck','MTruck','LTruck','Bus'])
classes = (['Sedan', 'Van', 'STruck', 'MTruck', 'LTruck', 'Bus','Traffic_Cone','Traffic_Bar','Traffic_Barrier'])
data_root = '/home/boden/Dev/ws/BDPilotDataset/BDMerge/'
# data_root = '/home/boden/Dev/ws/BDPilotDataset/LaneMultiDatasets'
data = dict(
    samples_per_gpu=gpus,
    workers_per_gpu=2,
    train=dict(
       type=dataset_type,
        classes = classes,
        ann_file='Annotations/BDPilot_annotations_crop_640_multi_img.json',
        # ann_file='BDPilot_annotations_LaneMultiDatasets.json',
        data_root = data_root,
        img_prefix = 'images_crop_640',
        # img_prefix='Images',
        lane_seg_prefix='GroundTruth/lane_crop_640',
        # lane_seg_prefix='GT',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes = classes,
        ann_file='Annotations/BDPilot_annotations_crop_640_multi_img.json',
        # ann_file='BDPilot_annotations_LaneMultiDatasets.json',
        data_root = data_root,
        img_prefix = 'images_crop_640',
        # img_prefix='Images',
        lane_seg_prefix='GroundTruth/lane_crop_640',
        # lane_seg_prefix='GT',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes = classes,
        ann_file='Annotations/BDPilot_annotations_crop_640_multi_img.json',
        # ann_file='BDPilot_annotations_LaneMultiDatasets.json',
        data_root = data_root,
        img_prefix = 'images_crop_640',
        # img_prefix='Images',
        lane_seg_prefix='GroundTruth/lane_crop_640',
        # lane_seg_prefix='GT',
        pipeline=test_pipeline))

optimizer = dict(type='Adam', lr=0.0002, betas=(0.9, 0.999), eps=1e-08)
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[45, 60],
    gamma=0.1,
)

checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
    ])


# yapf:enable
# runtime settings
total_epochs = 100
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/boden/Dev/ws/chenyuhao/mmdet_laneline/BDPilot_prune/model_prune/pruned_test.pth'
# load_from = '/home/boden/Dev/ws/ws/Lane_newdatasets/work_dirs/20210420/epoch_25.pth'
# load_from = None
# resume_from = '/home/boden/Dev/ws/chenyuhao/mmdet_laneline/Lane_newdatasets/work_dirs/20210402/without_neck_test/latest.pth'
resume_from = None

# work_dir = '/home/boden/Dev/ws/chenyuhao/mmdet_laneline/Lane_newdatasets/work_dirs/20210331/pretrain'
# work_dir = '/home/boden/Dev/ws/ws/Lane_newdatasets/work_dirs/20210420_2/'
work_dir = '/home/boden/Dev/ws/chenyuhao/mmdet_laneline/BDPilot_prune/work_dirs/20210610'

workflow = [('train', 1)]