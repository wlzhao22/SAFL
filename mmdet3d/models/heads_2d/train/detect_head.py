import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init
import numpy as np
import math
from mmdet.core import multi_apply
from mmdet3d.models.losses.losses_2d.focal_loss import ct_focal_loss
from mmdet.models.builder import HEADS
from abc import ABCMeta
from mmdet3d.models.utils.utils_2d.key_config import *
from mmdet3d.models.utils.utils_2d.gaussian_target import draw_umich_gaussian, gaussian_radius
from mmdet3d.models.utils.utils_2d.functional import tranpose_and_gather_feat
from mmdet3d.models.utils.utils_2d.dla_up import DLAUp

@HEADS.register_module()
class DetectHead(nn.Module,metaclass=ABCMeta):

    def __init__(self,
                 n_id = 5983,
                 num_classes=81,
                 wh_agnostic=True,
                 wh_gaussian=True,
                 hm_weight=2.,
                 wh_weight=0.2,
                 wheel_weight=5,
                 train_cfg=None,
                 test_cfg=None,
                 with_wheel=False,
                 with_tracking=False):
        super(DetectHead, self).__init__()

        self.num_classes = num_classes
        self.wh_agnostic = wh_agnostic
        self.wh_gaussian = wh_gaussian
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.with_tracking = with_tracking

        self.n_id = n_id
        self.id_dim = 128
        self.down_ratio = 4
        self.wh_planes = 2
        self.off2d_channel = 2
        self.wheel_channel = 4

        self.first_level = int(np.log2(4))
        channels = [16, 32, 64, 128, 256, 512]
        # channels = [16, 32, 64, 64, 128, 256]
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)

        self.with_wheel = with_wheel
        self.wheel_weight = wheel_weight

        # heads
        self.heads_channels = {}
        self.heads_channels[OUTPUT_DETECTION_OBJECT_HEATMAP] = self.num_classes
        self.heads_channels[OUTPUT_DETECTION_OBJECT_WH] = self.wh_planes
        self.heads_channels[OUTPUT_DETECTION_OBJECT_OFF2D] = self.off2d_channel
        
        if self.with_wheel:
            self.heads_channels[OUTPUT_DETECTION_OBJECT_WHEEL_OFF2D]=self.wheel_channel*2
            self.heads_channels[OUTPUT_DETECTION_WHEEL_HEATMAP]=self.wheel_channel
            self.heads_channels[OUTPUT_DETECTION_WHEEL_HEATMAP_OFF2D]=self.off2d_channel

        if self.with_tracking:
            self.emb_scale = math.sqrt(2) * math.log(self.n_id - 1)
            self.heads_channels[OUTPUT_DETECTION_ID] = self.id_dim
            self.classifier = nn.Linear(self.id_dim, self.n_id)

        for head in self.heads_channels:
            num_output = self.heads_channels[head]
            fc = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=(3, 1), padding=(1, 0), bias=True),
                nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1), bias=True),
                # nn.GroupNorm(2,32),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, num_output, kernel_size=1, stride=1, padding=0))
            self.__setattr__('{}'.format(head), fc)
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward_single(self, feats, depth=None):
        x = self.dla_up(feats[self.first_level:])
        ret = {}
        for head in self.heads_channels:
            fpn_out = self.__getattr__('{}'.format(head))(x)
            ret[head] = fpn_out

        return ret

    def forward(self, feats):

        return self.forward_single(feats)


    def forward_train(self, inputs, outputs):
        feature = outputs[OUTPUT_FEATURE]
        pred = self.forward_single(feature[0])
        outputs.update(pred)
        detect_loss = self.loss(inputs, outputs)

        return detect_loss

    def loss(self, inputs, outputs):
        inputs.update(self.target_generator(inputs, outputs))
        detection_loss = self.loss_calc(inputs, outputs)
        return detection_loss

    def target_generator(self, inputs, outputs):
        gt_bboxes_2d, gt_labels, img_metas = inputs[COLLECTION_GT_BBOX_2D], inputs[COLLECTION_GT_LABEL], inputs[COLLECTION_IMG_METAS]
        target_res = {}
        with torch.no_grad():
            if self.with_tracking:
                gt_reids = inputs[COLLECTION_GT_REID]
                heatmap, wh, ind, reg_mask, off2d, reid, id_mask = multi_apply(
                    self.target_single_image,
                    gt_bboxes_2d,
                    gt_labels,
                    img_metas,
                    gt_reids,
                )
                target_res[DETECTION_OBJECT_REID] = torch.stack(reid, dim=0).detach()
                target_res[DETECTION_OBJECT_ID_MASK] = torch.stack(id_mask, dim=0).detach()
            elif self.with_wheel:
                gt_wheels, gt_wheels_exist = inputs.get(COLLECTION_GT_WHEEL, None), inputs.get(COLLECTION_GT_WHEEL_EXIST, None)
                heatmap, wh, ind, reg_mask, off2d, hm_wheel_off_2d, hm_wheel_off_2d_ind, hm_wheel_off_2d_mask, wheel, wheel_mask, heatmap_wheel= multi_apply(
                    self.target_single_image,
                    gt_bboxes_2d,
                    gt_labels,
                    img_metas,
                    gt_wheels=gt_wheels,
                    gt_wheels_exist=gt_wheels_exist,
                )

                target_res[DETECTION_OBJECT_WHEEL_OFF2D] = torch.stack(hm_wheel_off_2d, dim=0).detach()
                target_res[DETECTION_OBJECT_WHEEL_IND] = torch.stack(hm_wheel_off_2d_ind, dim=0).detach()
                target_res[DETECTION_OBJECT_WHEEL_MASK] = torch.stack(hm_wheel_off_2d_mask, dim=0).detach()
                target_res[DETECTION_OBJECT_WHEEL_SUB] = torch.stack(wheel, dim=0).detach()
                target_res[DETECTION_WHEEL_MASK] = torch.stack(wheel_mask, dim=0).detach()
                target_res[DETECTION_WHEEL_HEATMAP] = torch.stack(heatmap_wheel, dim=0).detach()
            else:
                heatmap, wh, ind, reg_mask, off2d = multi_apply(
                    self.target_single_image,
                    gt_bboxes_2d,
                    gt_labels,
                    img_metas,
                )

            target_res[DETECTION_OBJECT_HEATMAP] = torch.stack(heatmap, dim=0).detach()
            target_res[DETECTION_OBJECT_WH] = torch.stack(wh,dim=0).detach()

            target_res[DETECTION_OBJECT_IND] = torch.stack(ind, dim=0).detach()
            target_res[DETECTION_OBJECT_MASK] = torch.stack(reg_mask, dim=0).detach()
            target_res[DETECTION_OBJECT_OFF2D] = torch.stack(off2d, dim=0).detach()

            return target_res

    def loss_calc(self, inputs, outputs):
        heatmap, wh, ind, reg_mask, off2d = inputs[DETECTION_OBJECT_HEATMAP],inputs[DETECTION_OBJECT_WH], inputs[DETECTION_OBJECT_IND], inputs[DETECTION_OBJECT_MASK], inputs[DETECTION_OBJECT_OFF2D]
        pred_hm, pred_wh, pred_off2d = outputs[OUTPUT_DETECTION_OBJECT_HEATMAP],outputs[OUTPUT_DETECTION_OBJECT_WH], outputs[OUTPUT_DETECTION_OBJECT_OFF2D]

        detection_loss = {}

        H, W = pred_hm.shape[2:]
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        detection_loss['hm_loss'] = ct_focal_loss(pred_hm, heatmap) * self.hm_weight

        pred_off2d = tranpose_and_gather_feat(pred_off2d,ind)
        mask_off2d = reg_mask.detach()
        mask_off2d = mask_off2d.unsqueeze(2).expand_as(pred_off2d).float()
        off2d_loss = F.l1_loss(pred_off2d * mask_off2d, off2d * mask_off2d, size_average=False)
        detection_loss['off2d_loss'] = off2d_loss / (mask_off2d.sum() + 1e-4)

        pred_wh = tranpose_and_gather_feat(pred_wh,ind)
        mask_wh = reg_mask.detach()
        mask_wh = mask_wh.unsqueeze(2).expand_as(pred_wh).float()
        wh_loss = F.l1_loss(pred_wh * mask_wh, wh * mask_wh, size_average=False)
        detection_loss['wh_loss'] = wh_loss * self.wh_weight / (mask_wh.sum() + 1e-4)

        if self.with_wheel:
            hm_wheel_off_2d, hm_wheel_off_2d_ind, hm_wheel_off_2d_mask, wheel, wheel_mask, heatmap_wheel = inputs[DETECTION_OBJECT_WHEEL_OFF2D],inputs[DETECTION_OBJECT_WHEEL_IND],inputs[DETECTION_OBJECT_WHEEL_MASK],inputs[DETECTION_OBJECT_WHEEL_SUB], inputs[DETECTION_WHEEL_MASK],inputs[DETECTION_WHEEL_HEATMAP]
            pred_wheel_off_2d, pred_hm_wheel, pred_hm_wheel_off_2d = outputs[OUTPUT_DETECTION_OBJECT_WHEEL_OFF2D], outputs[OUTPUT_DETECTION_WHEEL_HEATMAP], outputs[OUTPUT_DETECTION_WHEEL_HEATMAP_OFF2D]
            pred_wheel_off_2d = tranpose_and_gather_feat(pred_wheel_off_2d, ind)
            mask_off2d = wheel_mask.detach().float()
            mask_weight = mask_off2d.sum() + 1e-4
            detection_loss['wheel_off_2d_loss'] = self.wheel_weight * F.l1_loss(pred_wheel_off_2d*mask_off2d, wheel*mask_off2d, reduction='sum') / mask_weight

            pred_hm_wheel = torch.clamp(pred_hm_wheel.sigmoid_(), min=1e-4, max=1 - 1e-4)
            detection_loss['hm_wheel_loss'] = self.wheel_weight * ct_focal_loss(pred_hm_wheel, heatmap_wheel)

            pred_hm_wheel_off_2d = tranpose_and_gather_feat(pred_hm_wheel_off_2d, hm_wheel_off_2d_ind)
            hm_wheel_off_2d_mask = hm_wheel_off_2d_mask.detach()
            mask_weight = hm_wheel_off_2d_mask.sum() + 1e-4
            hm_wheel_off_2d_mask = hm_wheel_off_2d_mask.unsqueeze(2).expand_as(hm_wheel_off_2d).float()
            detection_loss['hm_wheel_off_2d_loss'] = self.wheel_weight * F.l1_loss(pred_hm_wheel_off_2d * hm_wheel_off_2d_mask, hm_wheel_off_2d * hm_wheel_off_2d_mask, reduction='sum') / mask_weight

        if self.with_tracking:
            pred_feature = outputs[OUTPUT_DETECTION_ID]
            reid = inputs[DETECTION_OBJECT_REID]
            id_mask = inputs[DETECTION_OBJECT_ID_MASK]
            pred_feature = tranpose_and_gather_feat(pred_feature, ind)
            pred_feature = pred_feature[id_mask>0].contiguous()
            pred_feature = self.emb_scale * F.normalize(pred_feature)
            id_target = reid[id_mask > 0]
            id_output = self.classifier(pred_feature).contiguous()
            idloss = nn.CrossEntropyLoss(ignore_index=-1)
            id_loss = idloss(id_output, id_target)
            detection_loss['tracking_loss'] = id_loss

        return  detection_loss

    def target_single_image(self, gt_bboxes_2d, gt_labels, img_metas, gt_reids=None, gt_wheels=None, gt_wheels_exist=None):

        output_h, output_w = img_metas['pad_shape'][0] // self.down_ratio, img_metas['pad_shape'][1] // self.down_ratio
        heatmap_channel = self.num_classes
        heatmap_wheel_channel = self.wheel_channel

        feat_gt_boxes = gt_bboxes_2d / self.down_ratio


        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                               max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                               max=output_h - 1)

        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])
        
        gt_wh = torch.stack([feat_ws,feat_hs],dim=1)
        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct = (torch.stack([(gt_bboxes_2d[:, 0] + gt_bboxes_2d[:, 2]) / 2,
                                (gt_bboxes_2d[:, 1] + gt_bboxes_2d[:, 3]) / 2],
                               dim=1) / self.down_ratio)

        ct_ints = (torch.stack([(gt_bboxes_2d[:, 0] + gt_bboxes_2d[:, 2]) / 2,
                                (gt_bboxes_2d[:, 1] + gt_bboxes_2d[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        num_objs = gt_labels.shape[0]
        max_obj = 100 #单张图片中最多多少个objuect
        ind = torch.zeros((max_obj,),dtype=torch.int64).to("cuda")
        reg_mask = torch.zeros((max_obj,),dtype=torch.uint8).to("cuda")
        off2d = torch.zeros((max_obj, 2), dtype=torch.float32).to("cuda")
        wh = torch.zeros((max_obj, 2), dtype=torch.float32).to("cuda")
        hm = torch.zeros((heatmap_channel, output_h, output_w), dtype=torch.float32).to("cuda")

        if self.with_tracking:
            ids = torch.zeros((max_obj,), dtype=torch.int64).to("cuda")
            id_mask = torch.zeros((max_obj,),dtype=torch.uint8).to("cuda")

        if self.with_wheel:
            key_pts = (gt_wheels / self.down_ratio)
            key_pts_ints = (gt_wheels / self.down_ratio).to(torch.int)
            wheel = torch.zeros((max_obj, heatmap_wheel_channel * 2), dtype=torch.float32).to("cuda")
            wheel_mask = torch.zeros((max_obj, heatmap_wheel_channel * 2), dtype=torch.uint8).to("cuda")
            hm_wheel = torch.zeros((heatmap_wheel_channel, output_h, output_w), dtype=torch.float32).to("cuda")
            hm_wheel_off_2d = torch.zeros((max_obj * heatmap_wheel_channel, 2), dtype=torch.float32).to("cuda")
            hm_wheel_off_2d_ind = torch.zeros((max_obj * heatmap_wheel_channel), dtype=torch.int64).to("cuda")
            hm_wheel_off_2d_mask = torch.zeros((max_obj * heatmap_wheel_channel), dtype=torch.int64).to("cuda")

        draw_gaussian = draw_umich_gaussian
        for k in range(num_objs):
            h, w = gt_wh[k,1], gt_wh[k,0]
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            cls_id = gt_labels[k]
            draw_gaussian(hm[cls_id], ct_ints[k], radius)
            ind[k] = ct_ints[k,1] * output_w + ct_ints[k,0]
            reg_mask[k] = 1
            off2d[k] = ct[k] - ct_ints[k]
            wh[k] = gt_wh[k]
            if self.with_wheel:
                for index in range(heatmap_wheel_channel):
                    if gt_wheels_exist[k, index] > 0:
                        hm_wheel_off_2d[k * heatmap_wheel_channel + index] = key_pts[k, index*2:index*2+2] - key_pts_ints[k, index*2:index*2+2]
                        hm_wheel_off_2d_ind[k * heatmap_wheel_channel + index] = key_pts_ints[k, index*2+1] * output_w + key_pts_ints[k, index*2]
                        hm_wheel_off_2d_mask[k * heatmap_wheel_channel + index] = 1
                        wheel[k, index*2:index*2+2] = key_pts_ints[k, index*2:index*2+2] - ct_ints[k]
                        wheel_mask[k, index*2:index*2+2] = 1
                        draw_gaussian(hm_wheel[index], key_pts_ints[k, index*2:index*2+2], radius)
            if self.with_tracking:
                ids[k] = gt_reids[k]
                if cls_id < 7:
                    id_mask[k] = 1

        if self.with_wheel:
            return hm, wh, ind, reg_mask, off2d, hm_wheel_off_2d, hm_wheel_off_2d_ind, hm_wheel_off_2d_mask, wheel, wheel_mask, hm_wheel
        elif self.with_tracking:
            return hm, wh, ind, reg_mask, off2d, ids, id_mask
        else:
            return hm, wh, ind, reg_mask, off2d

