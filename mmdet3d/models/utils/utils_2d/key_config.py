'''
Author: your name
Date: 2020-12-06 18:54:34
LastEditTime: 2021-02-26 16:14:01
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /BDPilot/mmdet/models/utils/key_config.py
'''

###########################INPUT############################
COLLECTION_CAR_MASK = "car_mask"
COLLECTION_IMG = "img"
COLLECTION_IMG_PRE = "img_pre"
COLLECTION_IMG_POST = "img_post"
COLLECTION_IMG_AUG = "img_aug"
COLLECTION_IMG_AFFINE = "img_affine"
COLLECTION_IMGS = [COLLECTION_IMG, COLLECTION_IMG_PRE, COLLECTION_IMG_POST]

COLLECTION_GT_BBOX_2D = "gt_bboxes_2d"
COLLECTION_GT_BBOX_3D = "gt_bboxes_3d"
COLLECTION_GT_LABEL = "gt_labels"
COLLECTION_GT_REID = "gt_reids"
COLLECTION_IMG_METAS = "img_metas"
COLLECTION_CALIB = "calib"
COLLECTION_INV_CALIB = "inv_calib"

COLLECTION_SCALED_CALIB = "scaled_calib"
COLLECTION_SCALED_INV_CALIB = "scaled_inv_calib"

COLLECTION_SCALED_CALIB_F = "scaled_calib_f"
COLLECTION_SCALED_INV_CALIB_F = "scaled_inv_calib_f"

COLLECTION_GT_SEM_SEG = "gt_semantic_seg"

COLLECTION_GT_LANE_SEG = "gt_lane_seg"
COLLECTION_GT_LANE_EXIST = "gt_lane_exist"
COLLECTION_GT_LANE_CLS_LABEL = 'gt_cls_label'
COLLECTION_GT_LANE_PTS = 'gt_lane_pts'

COLLECTION_GT_FREE_SPACE = "gt_free_space"
COLLECTION_GT_FREE_SPACE_RESIZED = "gt_free_space_resized"

COLLECTION_GT_WHEEL = "gt_wheels"
COLLECTION_GT_WHEEL_EXIST = "gt_wheels_exist"

COLLECTION_GT_DDD_HEAD_DIRECTION="gt_ddds_head_direction"
COLLECTION_GT_DDD_DX="gt_ddds_dx"
COLLECTION_GT_DDD_DW="gt_ddds_dw"
COLLECTION_GT_DDD_SIZES="gt_ddds_size"
COLLECTION_GT_DDD_ROTATIONS="gt_ddd_rotations"
COLLECTION_GT_DDD_CENTER_2D="gt_ddds_center_2d"
COLLECTION_GT_DDD_L0="gt_ddds_l0"
COLLECTION_GT_DDD_L1="gt_ddds_l1"
COLLECTION_GT_DDD_L2="gt_ddds_l2"

COLLECTION_GT_POSITION='position'

COLLECTION_IMG_REF = "img_c"
COLLECTION_GT_BBOX_2D_REF = "gt_bboxes_2d_c"
COLLECTION_GT_LABEL_REF = "gt_labels_c"
COLLECTION_GT_REID_REF = "gt_reids_c"
OUTPUT_TEACHER_FEATURE_REF = ("teacher_img_ref_feature", 0)
OUTPUT_TEACHER_BACKBONE_REF = ("teacher_img_ref_backbone", 0)
OUTPUT_STUDENT_BACKBONE_REF = ("student_img_ref_backbone", 0)
OUTPUT_STUDENT_FEATURE_REF = ("student_img_ref_feature", 0)
OUTPUT_DETECTION_YOLO_BOX_LIST_REF = "detection_bbox_list_ref"
OUTPUT_TRACK = "track"
OUTPUT_TEACHER_TRACK_EMBEDDING = "teacher_embedding"
OUTPUT_TEACHER_TRACK_EMBEDDING_REF = "teacher_embedding_ref"
TEACHER_HEADER_TRACK = "teacher_track_head"
STUDENT_HEADER_TRACK = "student_track_head"
OUTPUT_STUDENT_TRACK_EMBEDDING = "student_embedding"
OUTPUT_STUDENT_TRACK_EMBEDDING_REF = "student_embedding_ref"

WEIGHT_FIXED_PARAMETER = "fix_para"

DETECTION_OBJECT_HEATMAP = 'heatmap'
DETECTION_OBJECT_WH = 'wh'
DETECTION_OBJECT_IND = 'ind'
DETECTION_OBJECT_MASK = 'reg_mask'
DETECTION_OBJECT_ID_MASK = 'id_mask'
DETECTION_OBJECT_OFF2D = 'off2d'
DETECTION_OBJECT_REID = 'reid'
DETECTION_OBJECT_WHEEL_OFF2D = 'hm_wheel_off2d'
DETECTION_OBJECT_WHEEL_IND = 'hm_wheel_off2d_ind'
DETECTION_OBJECT_WHEEL_MASK = 'hm_wheel_off2d_mask'
DETECTION_OBJECT_WHEEL_SUB = 'wheel'
DETECTION_WHEEL_MASK = 'wheel_mask'
DETECTION_WHEEL_HEATMAP = 'heatmap_wheel'



###########################OUTPUT###########################
OUTPUT_BBOX = "bbox"
OUTPUT_DEPTH = "depth"
OUTPUT_BBOXES = "bboxes"
OUTPUT_IDS_FEATURE = "ids_feature"
OUTPUT_FEATURE = ("feature", 0)
OUTPUT_AUTO_IMG = "auto_img"
OUTPUT_DETECT = "detect"

OUTPUT_TEACHER_FEATURE = ("teacher_feature", 0)
OUTPUT_STUDENT_FEATURE = ("student_feature", 0)
OUTPUT_TEACHER_PRED = ("teacher_pred", 0)
OUTPUT_TEACHER_PRED_LANE = ("teacher_pred_lane",0)
OUTPUT_TEACHER_PRED_FREESPACE = ("teacher_pred_freespace",0)
OUTPUT_TEACHER_PRED_DETECTION = ("teacher_pred_detection",0)
OUTPUT_TEACHER_BACKBONE = ("teacher_backbone",0)
OUTPUT_STUDENT_BACKBONE = ("student_backbone",0)
OUTPUT_TEACHER_BACKBONE_AFFINE = ("teacher_backbone_affine",0)
OUTPUT_STUDENT_BACKBONE_AFFINE = ("student_backbone_affine",0)
OUTPUT_TEACHER_BACKBONE_AUG = ("teacher_backbone_aug",0)
OUTPUT_STUDENT_BACKBONE_AUG = ("student_backbone_aug",0)
OUTPUT_TEACHER_FEATURE_AFFINE = ("teacher_feature_affine", 0)
OUTPUT_STUDENT_FEATURE_AFFINE = ("student_feature_affine", 0)
OUTPUT_TEACHER_FEATURE_AUG = ("teacher_feature_aug", 0)
OUTPUT_STUDENT_FEATURE_AUG = ("student_feature_aug", 0)

OUTPUT_TEACHER_PRED_FREESPACE = ("teacher_pred_freespace",0)
OUTPUT_TEACHER_PRED_LANE = ("teacher_pred_lane",0)
OUTPUT_TEACHER_PRED_DETECTION = ("teacher_pred_detection",0)

OUTPUT_FEATURE_AFFINE = ("img_affine_feature", 0)
OUTPUT_BACKBONE_FEATURE_AFFINE = ("img_backbone_feature", 0)
OUTPUT_AUTO_FEATURE =("auto_feature", 0)
OUTPUT_FEATURE_AUG = ("feature_aug", 0)
OUTPUT_AUTO_FEATURE_PRE = ("auto_feature", -1)
OUTPUT_AUTO_FEATURE_POST =  ("auto_feature", 1)
OUTPUT_AUTO_FEATURES = [OUTPUT_AUTO_FEATURE, OUTPUT_AUTO_FEATURE_PRE, OUTPUT_AUTO_FEATURE_POST]

OUTPUT_AUTO_FEATURE_PRE_RESAMPLE =  ("resample_feature", -1)
OUTPUT_AUTO_FEATURE_POST_RESAMPLE =  ("resample_feature", 1)
OUTPUT_AUTO_FEATURE_RESAMPLES = [OUTPUT_AUTO_FEATURE_PRE_RESAMPLE, OUTPUT_AUTO_FEATURE_POST_RESAMPLE]

OUTPUT_AUTO_SCALE_0 = ("res", 0)
OUTPUT_AUTO_SCALE_1 = ("res", 1)
OUTPUT_AUTO_SCALE_2 = ("res", 2)
OUTPUT_AUTO_SCALE_3 = ("res", 3)
OUTPUT_AUTO_SCALES = [OUTPUT_AUTO_SCALE_0, OUTPUT_AUTO_SCALE_1, OUTPUT_AUTO_SCALE_2, OUTPUT_AUTO_SCALE_3]
OUTPUT_AUTO_IMG = "res_img"

OUTPUT_POSE_CUR_TO_PRE =  ("cam_T_cam", 0, -1)
OUTPUT_POSE_CUR_TO_POST =  ("cam_T_cam", 0, 1)
OUTPUT_POSES = [OUTPUT_POSE_CUR_TO_PRE, OUTPUT_POSE_CUR_TO_POST]

OUTPUT_DISP_SCALE_0 = ("disp", 0)
OUTPUT_DISP_SCALE_1 = ("disp", 1)
OUTPUT_DISP_SCALE_2 = ("disp", 2)
OUTPUT_DISP_SCALE_3 = ("disp", 3)
OUTPUT_DISP_SCALE_4 = ("disp", 4)
OUTPUT_DISP_SCALES = [OUTPUT_DISP_SCALE_0, OUTPUT_DISP_SCALE_1, OUTPUT_DISP_SCALE_2, OUTPUT_DISP_SCALE_3, OUTPUT_DISP_SCALE_4]

OUTPUT_LANE = "lane"
OUTPUT_LANE_SEG = "lane_seg"
OUTPUT_LANE_SEG_D16 = "lane_seg_d16"
OUTPUT_LANE_SEG_D32 = "lane_seg_d32"
OUTPUT_LANE_SEG_LIST = "lane_seg_list"
OUTPUT_LANE_EXIST = "lane_exist"
OUTPUT_LANE_SEG_X = "lane_seg_x"
OUTPUT_LANE_SEG_Y = "lane_seg_y"
OUTPUT_LANE_REG_XY = "lane_reg_xy"

OUTPUT_FREE_SPACE = "free_space_pred"

OUTPUT_KEYPOINT_HM = "keypoint_hm"
OUTPUT_KEYPOINT_OFF = "keypoint_off"
OUTPUT_KEYPOINT_FEATURE = "keypoint_feature"

OUTPUT_DETECTION_OBJECT_HEATMAP = 'hm'
OUTPUT_DETECTION_OBJECT_WH = 'wh'
OUTPUT_DETECTION_OBJECT_OFF2D = 'off_2d'
OUTPUT_DETECTION_OBJECT_WHEEL_OFF2D = 'wheel_off_2d'
OUTPUT_DETECTION_WHEEL_HEATMAP = 'hm_wheel'
OUTPUT_DETECTION_WHEEL_HEATMAP_OFF2D = 'hm_wheel_off_2d'
OUTPUT_DETECTION_ID = 'id'

OUTPUT_DETECTION_DD_REGRESS = "regress"
OUTPUT_DETECTION_DD_CLASSES = "classes"
OUTPUT_DETECTION_DD_ANCHORS = "anchors"
OUTPUT_DETECTION_DD_EMBEDDINGS = "embeddings"
OUTPUT_DETECTION_DDD_REGRESS = "ddd_regress"
OUTPUT_DETECTION_DDD_CLASSES = "ddd_classes"

OUTPUT_DETECTION_DDD_BBOX = "bbox"
OUTPUT_DETECTION_DDD_SCORE = "score"
OUTPUT_DETECTION_DDD_LABEL = "label"
OUTPUT_DETECTION_DDD_DW = "dw"
OUTPUT_DETECTION_DDD_DX = "dx"
OUTPUT_DETECTION_DDD_L0 = "l0"
OUTPUT_DETECTION_DDD_L1 = "l1"
OUTPUT_DETECTION_DDD_L2 = "l2"
OUTPUT_DETECTION_DDD_RES_X = "res_x"
OUTPUT_DETECTION_DDD_RES_Y = "res_y"
OUTPUT_DETECTION_DDD_H = "h"
OUTPUT_DETECTION_DDD_W = "w"
OUTPUT_DETECTION_DDD_L = "l"
OUTPUT_DETECTION_DDD_HEAD_LABEL = "head_label"
OUTPUT_DETECTION_DDD_HEAD_SCORE = "head_score"
OUTPUT_DETECTION_DDD_ROTATION_CENTER = "rotation_center"
OUTPUT_DETECTION_DDD_ROTATION_RES = "rotation_res"

OUTPUT_DETECTION_YOLO_CONTENT = "detection_content"
OUTPUT_DETECTION_YOLO_BOX_LIST = "detection_bbox_list"

OUTPUT_DETECTION_STATIC_BBOX = "static_detect"
###########################HEADER###########################
HEADER_DYNAMIC_BBOX = "dynamic_bbox_head"
HEADER_DEPTH = "depth_head"
HEADER_LANE = "lane_head"
HEADER_AUTO = "auto_head"
HEADER_FREE_SPACE = "free_space_head"
HEADER_STATIC_BBOX = "static_bbox_head"
HEADER_LANE_FREESPACE = "lane_freespace_head"
#########################BACKBONE###########################
BACKBONE_FEATURE = "feature_backbone"
BACKBONE_AUTO = "auto_backbone"
BACKBONE_POSE = "pose_backbone"
