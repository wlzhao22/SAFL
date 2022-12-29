from ...builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from ...utils.utils_2d.key_config import *



@DETECTORS.register_module()
class BDNetDistill(BaseDetector):
    """Base class for tracking detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck and tracking instances based on the embeddings of output.

    """

    def __init__(self,
                 collect_keys=["img", "img_metas", "gt_bboxes_2d", \
                               "gt_bboxes_3d", "gt_labels", "gt_reids", \
                               "calib", "img_aug", "img_pre", "img_post", \
                               "car_mask", "inv_calib", "gt_lane_seg", "gt_lane_exist"],
                 teacher=None,
                 student=None,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None,
                 train_student=False):
        super(BDNetDistill, self).__init__()
        super(BDNetDistill, self).init_weights(pretrained)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.collect_keys = collect_keys
        self.teacher = teacher
        self.student = student
        self.pretrained = pretrained
        self.train_student = train_student

        self.init_nets()

        
    def init_nets(self):

        for k,v in self.teacher.backbones.items():
            if v is not None:
                bk = build_backbone(v)
                k = 'teacher_'+k
                setattr(self,k,bk)

        for k,v in self.teacher.headers.items():
            if v is not None:     
                hd = build_head(v)
                k = 'teacher_'+k
                setattr(self,k,hd)
        
        for k,v in self.teacher.necks.items():
            if v is not None:     
                ne = build_neck(v)
                k = 'teacher_'+k
                setattr(self,k,ne)

        if self.train_student:
            for k,v in self.student.backbones.items():
                if v is not None:
                    bk = build_backbone(v)
                    k = 'student_'+k
                    setattr(self,k,bk)

            for k,v in self.student.headers.items():
                if v is not None:     
                    hd = build_head(v)
                    k = 'student_'+k
                    setattr(self,k,hd)

            for k,v in self.student.necks.items():
                if v is not None:     
                    ne = build_neck(v)
                    k = 'student_'+k
                    setattr(self,k,ne)
    
    def extract_feat(self, img):

        teacher_feat = self.teacher_feature_backbone(img)
        if self.train_student:
            student_feat = self.student_feature_backbone(img)
            return teacher_feat, student_feat
        else:
            return teacher_feat
    
    def fusion_feat_teacher(self, feature):
        if getattr(self, 'teacher_fpn', None):
            teacher_feat = self.teacher_fpn(feature)
        if getattr(self, 'teacher_fpn_env', None):
            teacher_feat_env = self.teacher_fpn_env(feature)

        return teacher_feat, teacher_feat_env

           
    def fusion_feat_student(self, feature):
        if getattr(self, 'student_fpn', None):
            student_feat = self.student_fpn(feature)
        if getattr(self, 'student_fpn_env', None):
            student_feat_env = self.student_fpn_env(feature)

        return student_feat, student_feat_env

    def init_inputs(self, imgs, img_metas, para):
        inputs = {}
        inputs[COLLECTION_IMG] = imgs
        inputs[COLLECTION_IMG_METAS] = img_metas
        for k, v in para.items():
            if k == COLLECTION_CAR_MASK:
                inputs[k] = (v < 200).float().detach().view(-1, 1, *v.shape[1:])
            else:
                inputs[k] = v

        return inputs

    def init_outputs(self, inputs):
        img = inputs[COLLECTION_IMG]
        img_aug = inputs[COLLECTION_IMG_AUG]
        img_affine = inputs[COLLECTION_IMG_AFFINE]
        outputs = {}
        
        if self.train_student:
            # FreeSpace
            outputs[OUTPUT_TEACHER_BACKBONE], outputs[OUTPUT_STUDENT_BACKBONE] = self.extract_feat(img)
            # Detection
            outputs[OUTPUT_TEACHER_BACKBONE_AFFINE], outputs[OUTPUT_STUDENT_BACKBONE_AFFINE] = self.extract_feat(img_affine)
            outputs[OUTPUT_TEACHER_FEATURE_AFFINE], _ = self.fusion_feat_teacher(outputs[OUTPUT_TEACHER_BACKBONE_AFFINE])
            outputs[OUTPUT_STUDENT_FEATURE_AFFINE], _ = self.fusion_feat_student(outputs[OUTPUT_STUDENT_BACKBONE_AFFINE])
            # Lane
            outputs[OUTPUT_TEACHER_BACKBONE_AUG], outputs[OUTPUT_STUDENT_BACKBONE_AUG] = self.extract_feat(img_aug)
            _, outputs[OUTPUT_TEACHER_FEATURE_AUG] = self.fusion_feat_teacher(outputs[OUTPUT_TEACHER_BACKBONE_AUG])
            _, outputs[OUTPUT_STUDENT_FEATURE_AUG] = self.fusion_feat_student(outputs[OUTPUT_STUDENT_BACKBONE_AUG])
            # Track
            outputs[OUTPUT_TEACHER_BACKBONE_REF], outputs[OUTPUT_STUDENT_BACKBONE_REF] = self.extract_feat(inputs[COLLECTION_IMG_REF])
            outputs[OUTPUT_TEACHER_FEATURE_REF],_ = self.fusion_feat_teacher(outputs[OUTPUT_TEACHER_BACKBONE_REF])
            outputs[OUTPUT_STUDENT_FEATURE_REF],_ = self.fusion_feat_student(outputs[OUTPUT_STUDENT_BACKBONE_REF])

        else:
            # FreeSpace
            outputs[OUTPUT_TEACHER_BACKBONE] = self.extract_feat(img)

            # Detection
            outputs[OUTPUT_TEACHER_BACKBONE_AFFINE] = self.extract_feat(img_affine)
            outputs[OUTPUT_TEACHER_FEATURE_AFFINE], _ = self.fusion_feat_teacher(outputs[OUTPUT_TEACHER_BACKBONE_AFFINE])

            # Lane
            outputs[OUTPUT_TEACHER_BACKBONE_AUG] = self.extract_feat(img_aug)
            _, outputs[OUTPUT_TEACHER_FEATURE_AUG] = self.fusion_feat_teacher(outputs[OUTPUT_TEACHER_BACKBONE_AUG])

            # Track
            outputs[OUTPUT_TEACHER_BACKBONE_REF] = self.extract_feat(inputs[COLLECTION_IMG_REF])
            outputs[OUTPUT_TEACHER_FEATURE_REF],_ = self.fusion_feat_teacher(outputs[OUTPUT_TEACHER_BACKBONE_REF])


        return outputs

    def compute_losses(self, inputs, outputs):
        losses = {}
        if self.train_student:
            for k, v in self.teacher.headers.items():
                if v is not None:
                    k = 'teacher_' + k
                    if k == 'teacher_free_space_head':
                        outputs = getattr(self, k).forward_teaching(outputs)

            for k, v in self.student.headers.items():
                if v is not None:
                    k = 'student_' + k
                    loss = getattr(self, k).forward_train(inputs, outputs)
                    losses.update(loss)
        else:
            for k, v in self.teacher.headers.items():
                if v is not None:
                    k = 'teacher_' + k
                    loss = getattr(self, k).forward_train(inputs, outputs)
                    losses.update(loss)

        return losses

    def forward_dummy(self, img):
        outputs = {}
        if self.train_student:
            if getattr(self, 'student_feature_backbone', None):
                x = self.extract_feat(img)
                x_nk = self.fusion_feat_student(x[1])

            if getattr(self, 'student_lane_head', None):
                outputs[OUTPUT_LANE]  = self.student_lane_head(x_nk[1])

            if getattr(self, 'student_free_space_head', None):
                outputs[OUTPUT_FREE_SPACE] = self.student_free_space_head(x[1])

            if getattr(self, 'student_dynamic_bbox_head', None):
                detection_bbox_list = self.student_dynamic_bbox_head(x_nk[0])[OUTPUT_DETECTION_YOLO_BOX_LIST]
                outputs[OUTPUT_DETECT] = self.student_dynamic_bbox_head.get_nms_result(detection_bbox_list)

            if getattr(self, 'student_static_bbox_head', None):
                outputs[OUTPUT_DETECTION_STATIC_BBOX] = self.student_static_bbox_head(x_nk[0])

        else:
            if getattr(self, 'teacher_feature_backbone', None):
                x = self.extract_feat(img)
                x1 = self.fusion_feat_teacher(x)
            if getattr(self, 'teacher_lane_head', None):
                outputs[OUTPUT_LANE]  = self.teacher_lane_head(x1[1])
            
            if getattr(self, 'teacher_free_space_head', None):
                outputs[OUTPUT_FREE_SPACE] = self.teacher_free_space_head(x)

            if getattr(self, 'teacher_dynamic_bbox_head', None):
                detection_bbox_list = self.teacher_dynamic_bbox_head(x1[0])[OUTPUT_DETECTION_YOLO_BOX_LIST]
                outputs[OUTPUT_DETECT] = self.teacher_dynamic_bbox_head.get_nms_result(detection_bbox_list)

            if getattr(self, 'teacher_static_bbox_head', None):
                outputs[OUTPUT_DETECTION_STATIC_BBOX] = self.teacher_static_bbox_head(x1[0])

            
        return outputs

    def forward_train(self, imgs, img_metas, **kwargs):

        inputs = self.init_inputs(imgs, img_metas, kwargs)

        outputs = self.init_outputs(inputs)

        losses = self.compute_losses(inputs, outputs)

        return losses

    def simple_test(self, img, img_metas=None, rescale=False):
        outputs = {}
        outputs[COLLECTION_IMG_METAS] = img_metas

        if self.train_student:
            if getattr(self, 'student_feature_backbone', None):
                x = self.extract_feat(img)
                x_nk = self.fusion_feat_student(x[1])
            if getattr(self, 'student_lane_head', None):
                outputs[OUTPUT_LANE]  = self.student_lane_head(x_nk[1])
            if getattr(self, 'student_free_space_head', None):
                outputs[OUTPUT_FREE_SPACE] = self.student_free_space_head(x[1])
            # if getattr(self, 'student_fusion_head', None):
            #     outputs[OUTPUT_FREE_SPACE],outputs[OUTPUT_LANE] = self.student_fusion_head(x[1])
            if getattr(self, 'student_dynamic_bbox_head', None):
                detection_bbox_list = self.student_dynamic_bbox_head(x_nk[0])[OUTPUT_DETECTION_YOLO_BOX_LIST]
                outputs[OUTPUT_DETECT] = self.student_dynamic_bbox_head.get_nms_result(detection_bbox_list)
            if getattr(self, 'student_static_bbox_head', None):
                outputs[OUTPUT_DETECTION_STATIC_BBOX] = self.student_static_bbox_head(x_nk[0])   
            if getattr(self, "student_track_head", None):
                outputs[OUTPUT_TRACK] = self.student_track_head(x_nk[0])

        else:
            if getattr(self, 'teacher_feature_backbone', None):
                x = self.extract_feat(img)
                x1 = self.fusion_feat_teacher(x)
            if getattr(self, 'teacher_lane_head', None):
                outputs[OUTPUT_LANE]  = self.teacher_lane_head(x1[1])
            
            if getattr(self, 'teacher_free_space_head', None):
                outputs[OUTPUT_FREE_SPACE] = self.teacher_free_space_head(x)
                # print(outputs[OUTPUT_FREE_SPACE])
            
            if getattr(self, 'teacher_fusion_head', None):
                outputs[OUTPUT_FREE_SPACE],outputs[OUTPUT_LANE] = self.teacher_fusion_head(x)

            if getattr(self, 'teacher_dynamic_bbox_head', None):
                detection_bbox_list = self.teacher_dynamic_bbox_head(x1[0])[OUTPUT_DETECTION_YOLO_BOX_LIST]
                outputs[OUTPUT_DETECT] = self.teacher_dynamic_bbox_head.get_nms_result(detection_bbox_list)
            
            if getattr(self, 'teacher_static_bbox_head', None):
                outputs[OUTPUT_DETECTION_STATIC_BBOX] = self.teacher_static_bbox_head(x1[0])
            
            if getattr(self, "teacher_track_head", None):
                outputs[OUTPUT_TRACK] = self.teacher_track_head(x1[0])


        return outputs

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        raise NotImplementedError

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
            pass