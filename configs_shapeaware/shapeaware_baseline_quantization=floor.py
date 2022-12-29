

height = 384
width = 1280
mean = [124.16, 116.73, 103.936]
std = [58.624, 57.344, 57]
down_ratio = 4

collect_keys=['img', "gt_bboxes_2d", "gt_labels", 
    'gt_bboxes_3d', 'gt_labels_3d', 'flip',
    'edge_len', 'edge_indices',
]
meta_keys=(
    'filename', 'ori_filename', 
    # 'ori_shape',
    'img_info',
    'img_shape', 'pad_shape', 
    'img_norm_cfg',
    'calib'
)

# switches for ablation study 
aligned_nms = False
center_sampling = False
multi_center = False
lazy_regression = False
occlusion_level_pred = False
iou_score = False

num_classes = 3
class_names = ["Car", "Pedestrian", "Cyclist"]
train_cfg = None
test_cfg = dict(
    topk=100, local_maximum_kernel=3,
    max_per_img=50,
    nms_cfg=dict(
        nms_thr=0.0,
        score_threshold=0.0,
        use_rotate_nms=True,
    ),
    score_threshold=0.2, 
    relative_offset=(0, 0., 0),
    eval_dis_ious=False,
    eval_depth=False,
    pred_2d=True,
    uncertainty_as_confidence=True,
    max_depth=1e8,
)
model = dict(
    type='MonoFlex',
    backbone_cfg=dict(
        type='CustomDLASeg',
        base_name='dla34',    
        pretrained=True,  
        down_ratio=down_ratio,
        last_level=5,
        use_dcn=True, 
    ),
    neck_cfg=None,
    head_cfg=dict(
        type='MonoFlexCenterHead',
        input_width=width,
        input_height=height, 
        down_ratio=down_ratio,
        in_channels=64,
        max_objs=40,
        num_classes=num_classes,
        orientation='multi-bin',
        orientation_bin_size=4,
        consider_outside_objs=True,
        enable_edge_fusion=True,
        edge_fusion_norm='BN',
        keypoint_visible_modify=True,
        modify_invalid_keypoint_depth=True,
        heatmap_center='3D',
        approx_3d_center='intersect',
        adjust_boundary_heatmap=True,
        truncation_offset_loss='log',
        heatmap_ratio=0.5,
        reg_heads=[['2d_dim'], ['3d_offset'], ['corner_offset'], ['corner_uncertainty'], ['3d_dim'], ['ori_cls', 'ori_offset'], ['depth'], ['depth_uncertainty']],
        reg_channels=[[4, ], [2, ], [20], [3], [3, ], [8, 8], [1, ], [1, ]],
        corner_loss_depth='soft_combine',
        loss_names=['hm_loss', 'bbox_l1_loss', 'bbox_iou_loss', 'depth_loss', 'offset_loss', 'orien_loss', 'dims_loss', 'corner_loss', 'keypoint_loss', 'keypoint_depth_loss', 'trunc_offset_loss', 'weighted_avg_depth_loss'],
        init_loss_weight=[1, 0.01, 1, 1, 0.5, 1, 1, 0.2, 1.0, 0.2, 0.1, 0.2],
        loss_types=["Penalty_Reduced_FocalLoss", "L1", "giou", "L1"],
        center_mode='max',
        heatmap_type='centernet',
		dim_reg=['exp', True, False],
        depth_mode='inv_sigmoid',
        depth_output='soft',
        dim_weight=[1., 1., 1.],
        uncertainty_init=True,

        aligned_nms=aligned_nms,
        center_sampling=center_sampling,
        multi_center=multi_center, 
        lazy_regression=lazy_regression,
        occlusion_level_pred=occlusion_level_pred,
        iou_score=iou_score,
        quantization_method='floor',

        test_cfg=test_cfg
    )
)

# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
file_client_args = dict(backend='disk')

img_norm_cfg = dict(
    mean=mean, std=std, to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        with_occluded=False,
        file_client_args=file_client_args),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomFlip3D', sync_2d=True, flip_ratio_bev_horizontal=0.5),
    dict(type='Pad', size=(height, width)),
    dict(type='GetEdgeIndices', 
        input_width=width,
        input_height=height, 
        down_ratio=down_ratio),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
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
            dict(type='GetEdgeIndices', 
                input_width=width,
                input_height=height, 
                down_ratio=down_ratio),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'edge_len', 'edge_indices',], meta_keys=meta_keys),
        ])
]
debug_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(width, height),
        flip=False,
        transforms=[
            dict(type='Pad', size=(height, width)),
            dict(type='GetEdgeIndices',
                input_width=width,
                input_height=height,  
                down_ratio=down_ratio),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle3D', class_names=class_names),
            dict(type='Collect3D',
                keys=collect_keys,
                meta_keys=meta_keys
            ),
        ])
]

input_modality = dict(use_lidar=True, use_camera=True)
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/kitti_infos_train.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False,
        box_type_3d='camera'
    ),
    train_debug=dict(  # TODO: delete this
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/kitti_infos_train.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=debug_pipeline,
        modality=input_modality,
        classes=class_names,
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
        classes=class_names,
        test_mode=True,
        box_type_3d='camera'),
    val_debug=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=debug_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False,
        box_type_3d='camera'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/kitti_infos_test.pkl',
        split='testing',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='camera'),
    trainval=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/kitti_infos_trainval.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False,
        box_type_3d='camera'
    ),
)

total_epochs = 200
# optimizer = dict(type='Adam', lr=3e-4, betas=(0.9, 0.999), eps=1e-08)
# optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=1e-5)
optimizer = dict(type='AdamW', lr=3e-4, weight_decay=1e-5)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    step=[total_epochs - 10, total_epochs - 5],
    gamma=0.1,
)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

checkpoint_config = dict(interval=10)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)

dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = None
resume_from = None

workflow = [('train', 1)]
evaluation = dict(interval=1)
find_unused_parameters = True
