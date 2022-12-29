from ...builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from ...utils.utils_2d.key_config import *


 
@DETECTORS.register_module()
class BDNetFusion(BaseDetector):
    """Base class for tracking detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck and tracking instances based on the embeddings of output.

    """

    def __init__(self,
                 collect_keys=["img", "img_metas", "gt_bboxes_2d", \
                               "gt_bboxes_3d", "gt_labels", "gt_reids", \
                               "calib", "img_aug", "img_pre", "img_post", \
                               "car_mask", "inv_calib", "gt_lane_seg", "gt_lane_exist"],
                 backbones=None,
                 necks=None,
                 headers=None,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None):
        super(BDNetFusion, self).__init__()
        super(BDNetFusion, self).init_weights(pretrained)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.collect_keys = collect_keys
        self.backbones, self.necks, self.headers = backbones, necks, headers
        self.pretrained = pretrained

        self.init_nets()

    def init_nets(self):

        for k, v in self.backbones.items():
            if v is not None:
                bk = build_backbone(v)

                # init weight
                # bk.init_weights(pretrained=self.pretrained)
                #
                # # if backbone has key word of "fix_para", this network weight will be fixed
                # # and won't be trained
                # if v.get(WEIGHT_FIXED_PARAMETER, False):
                #     for p in bk.parameters():
                #         p.requires_grad = False

                setattr(self, k, bk)

        if self.necks is not None:
            for k, v in self.necks.items():
                if v is not None:
                    n = build_neck(v)

                    # init weight
                    # if isinstance(n, nn.Sequential):
                    #     for m in n:
                    #         m.init_weights(pretrained=self.pretrained)
                    # else:
                    #     n.init_weights(pretrained=self.pretrained)

                    # if neck has key word of "fix_para", this network weight will be fixed
                    # and won't be trained
                    # if v.get(WEIGHT_FIXED_PARAMETER, False):
                    #     for p in n.parameters():
                    #         p.requires_grad = False

                    setattr(self, k, n)

        for k, v in self.headers.items():
            if v is not None:
                h = build_head(v)
                # names[k].update(train_cfg=self.train_cfg, test_cfg=self.test_cfg)
                # init weight
                # h.init_weights()

                # if head has key word of "fix_para", this network weight will be fixed
                # and won't be trained
                # if v.get(WEIGHT_FIXED_PARAMETER, False):
                #     for p in h.parameters():
                #         p.requires_grad = False

                setattr(self, k, h)

    def extract_feat(self, img):
        p2, p3, p4, p5 = self.feature_backbone(img)
        features = (p2, p3, p4, p5)
        if self.necks:
            features_necks = self.bifpn(features)
            # features_neck_envs = self.bifpn_env(features)
            features_neck_envs = features_necks
        else:
            features_necks = features
            features_neck_envs = features
        return features, features_necks, features_neck_envs

    def extract_auto_feat(self, img):
        x = self.auto_backbone(img)
        return x

    def extract_pose(self, img_pre, img):
        pose_feature = self.pose_backbone(torch.cat([img_pre, img], 1))
        axisangle, translation = self.pose_head(pose_feature)
        return axisangle, translation

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
        if inputs.get(COLLECTION_IMG_PRE, None) is None:
            img, img_aug = inputs[COLLECTION_IMG], inputs[COLLECTION_IMG_AUG]
        else:
            img, img_aug, img_pre, img_post = inputs[COLLECTION_IMG], inputs[COLLECTION_IMG_AUG], \
                                     inputs[COLLECTION_IMG_PRE], inputs[COLLECTION_IMG_POST]
        outputs = {}

        if getattr(self, BACKBONE_FEATURE, None):
            # Freespace
            # outputs[OUTPUT_BACKBONE_FEATURE_AFFINE], _, outputs[OUTPUT_FEATURE], = self.extract_feat(img)
            
            # Detection
            outputs[OUTPUT_FEATURE_AFFINE] = self.extract_feat(img)[1]
            # outputs[OUTPUT_FEATURE_AFFINE] = self.extract_feat(inputs[COLLECTION_IMG_AFFINE])[1]
            # outputs[OUTPUT_FEATURE_REF] = self.extract_feat(inputs[COLLECTION_IMG_REF])[1]
            
            # Lane
            # outputs[OUTPUT_FEATURE_AUG] = self.extract_feat(img_aug)[2]

        if getattr(self, BACKBONE_POSE, None):
            axisangle_1, translation_1 = self.extract_pose(img_pre, img)
            axisangle_2, translation_2 = self.extract_pose(img, img_post)

            outputs[OUTPUT_POSE_CUR_TO_PRE] = self.pose_head.transformation_from_parameters(axisangle_1[:, 0],
                                                                                            translation_1[:, 0],
                                                                                            invert=True)
            outputs[OUTPUT_POSE_CUR_TO_POST] = self.pose_head.transformation_from_parameters(axisangle_2[:, 0],
                                                                                             translation_2[:, 0],
                                                                                         invert=False)

        return outputs

    def compute_losses(self, inputs, outputs):
        losses = {}
        for k, v in self.headers.items():
            if v is not None:
                loss = getattr(self, k).forward_train(inputs, outputs)
                losses.update(loss)

        return losses

    def forward_dummy(self, img):
        outputs = {}
        x = self.extract_feat(img)

        # if self.depth_head is not None:
        #     outputs.update(self.depth_head(x))

        # if getattr(self, HEADER_BBOX, None):
        #     outputs[OUTPUT_BBOX] = self.bbox_head(x)

        if getattr(self, HEADER_FREE_SPACE, None):
            outputs[OUTPUT_FREE_SPACE] = self.free_space_head(x[0])

        # if getattr(self, HEADER_LANE, None):
        #     outputs[OUTPUT_LANE]  = self.lane_head(x[0])
        
        # if getattr(self, "keypoint_head", None):
        #     outputs["keypoint"] = self.keypoint_head(x[0])

        if getattr(self, HEADER_DYNAMIC_BBOX, None):
            detection_bbox_list = self.dynamic_bbox_head(x[1])[OUTPUT_DETECTION_YOLO_BOX_LIST]
            outputs[OUTPUT_DETECT] = self.dynamic_bbox_head.get_nms_result(detection_bbox_list)
            
        if getattr(self, HEADER_STATIC_BBOX, None):
            outputs[OUTPUT_DETECTION_STATIC_BBOX] = self.static_bbox_head(x[1])

        # if getattr(self, HEADER_LANE_FREESPACE, None):
        #     outputs = self.lane_freespace_head(x[0])

        if getattr(self, "lane_head", None):
            outputs[OUTPUT_LANE]  = self.lane_head(x[2])

        if getattr(self, "track_head", None):
            outputs["track"] = self.track_head(x[1])

        return outputs

    def forward_train(self, imgs, img_metas, **kwargs):

        inputs = self.init_inputs(imgs, img_metas, kwargs)

        outputs = self.init_outputs(inputs)

        losses = self.compute_losses(inputs, outputs)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        outputs = {}
        outputs[COLLECTION_IMG_METAS] = img_metas
        if getattr(self, BACKBONE_FEATURE, None):
            x = self.extract_feat(img)

        # if getattr(self, HEADER_DEPTH, None):
        #     outputs.update(self.depth_head(x_depth))

        # if getattr(self, HEADER_LANE, None):
        #     outputs.update(self.lane_head(x[2]))

        # if getattr(self, HEADER_AUTO, None):
        #     # x = self.extract_auto_feat(img)
        #     outputs.update(self.auto_head(x_depth))

        # if getattr(self, "keypoint_head", None):
        #     outputs["keypoint"] = self.keypoint_head(x[0])
        
        # if getattr(self, 'bbox_side_head', None):
        #     detection_bbox_list = self.bbox_side_head(x)[OUTPUT_DETECTION_YOLO_BOX_LIST]
        #     outputs['bbox_side_detect'] = self.bbox_side_head.get_nms_result(detection_bbox_list)
            
        # if getattr(self, HEADER_FREE_SPACE, None):
        #     outputs[OUTPUT_FREE_SPACE] = self.free_space_head(x[0])
        
        if getattr(self, HEADER_LANE, None):
            outputs[OUTPUT_LANE]  = self.lane_head(x[2])

        # if getattr(self, HEADER_DYNAMIC_BBOX, None):
        #     detection_bbox_list = self.dynamic_bbox_head(x[1])[OUTPUT_DETECTION_YOLO_BOX_LIST]
        #     outputs[OUTPUT_DETECT] = self.dynamic_bbox_head.get_nms_result(detection_bbox_list)
            
        # if getattr(self, HEADER_STATIC_BBOX, None):
        #     outputs[OUTPUT_DETECTION_STATIC_BBOX] = self.static_bbox_head(x[1])

        # Todo
        # if getattr(self, HEADER_TRACK, None):
        #     outputs[OUTPUT_TRACK] = self.track_head(x[1])

        # if getattr(self, HEADER_LANE_FREESPACE, None):
        #      outputs.update(self.lane_freespace_head(x[1]))

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
