import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmdet3d.models.utils.utils_2d.efficientdet_utils import tranpose_and_gather_feat, ConvBlock
from mmdet3d.models.utils.utils_2d.gaussian_target import draw_umich_gaussian, gaussian_radius
from mmdet3d.models.losses.losses_2d.efficientdet_loss import calc_iou, ct_focal_loss
from mmdet3d.models.utils.utils_2d.key_config import *
from mmdet3d.models.builder import HEADS


class AnchorFreeModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AnchorFreeModule, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, out_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, U):
        f = self.conv(U)
        return f

@HEADS.register_module()
class CenterNetHeadTea(nn.Module):
    def __init__(self, in_channels, down_ratio=8, start_index=6, num_hm_cls=3, alpha=0.25, gamma=2.0):
        super(CenterNetHeadTea, self).__init__()
        # self.cSE = cSE(in_channels)
        self.hm = AnchorFreeModule(in_channels, num_hm_cls)
        self.wh = AnchorFreeModule(in_channels, 2)
        self.off = AnchorFreeModule(in_channels, 2)
        self.down_ratio = down_ratio
        self.start_index = start_index
        self.alpha = alpha
        self.gamma = gamma

    def forward_single(self, features):
        outputs = {}
        # f = self.cSE(features[0])
        f = features[0]
        outputs[OUTPUT_DETECTION_OBJECT_HEATMAP] = self.hm(f)
        outputs[OUTPUT_DETECTION_OBJECT_WH] = self.wh(f)
        outputs[OUTPUT_DETECTION_OBJECT_OFF2D] = self.off(f)

        return outputs
    
    def forward(self, features):
        return self.forward_single(features)
    
    def forward_train(self, inputs, outputs):
        # outputs.update(self.forward_single(outputs[OUTPUT_FEATURE]))
        outputs.update(self.forward_single(outputs[OUTPUT_TEACHER_FEATURE_AFFINE]))
        detect_loss = self.loss(inputs, outputs)

        return detect_loss
    
    def loss(self, inputs, outputs):
        gt_bboxes_2d, gt_labels = inputs[COLLECTION_GT_BBOX_2D], inputs[COLLECTION_GT_LABEL]
        barrier_hms, barrier_whs, barrier_offs = outputs[OUTPUT_DETECTION_OBJECT_HEATMAP], outputs[
            OUTPUT_DETECTION_OBJECT_WH], outputs[OUTPUT_DETECTION_OBJECT_OFF2D]
        barrier_hms = torch.clamp(barrier_hms.sigmoid_(), min=1e-4, max=1 - 1e-4)
       
        loss_dict = {}

        batch_size = barrier_hms.shape[0]
        static_det_losses = []
        hm_loss_list = []
        off2d_loss_list = []
        wh_loss_list = []

        for j in range(batch_size):
            gt_label = gt_labels[j]
            gt_bbox_2d = gt_bboxes_2d[j]

            static_idx = gt_label >= self.start_index
            static_bbox_2d = gt_bbox_2d[static_idx]
            static_label = gt_label[static_idx]

            # calculate loss for static objects
            barrier_hm = barrier_hms[j]
            barrier_wh = barrier_whs[j]
            barrier_off = barrier_offs[j]
            hm_cls, output_h, output_w = barrier_hm.shape
            feat_gt_boxes = static_bbox_2d / self.down_ratio
            feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0, max=output_w - 1)
            feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,  max=output_h - 1)
            feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1], feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])
            gt_wh = torch.stack([feat_ws, feat_hs], dim=1)
            ct = (torch.stack([(static_bbox_2d[:, 0] + static_bbox_2d[:, 2]) / 2,
                               (static_bbox_2d[:, 1] + static_bbox_2d[:, 3]) / 2],
                              dim=1) / self.down_ratio)

            ct_ints = (torch.stack([(static_bbox_2d[:, 0] + static_bbox_2d[:, 2]) / 2,
                                    (static_bbox_2d[:, 1] + static_bbox_2d[:, 3]) / 2],
                                   dim=1) / self.down_ratio).to(torch.int)
            num_objs = static_label.shape[0]
            max_obj = 100  # 单张图片中最多多少个objuect
            ind_ = torch.zeros((max_obj,), dtype=torch.int64).to("cuda")
            reg_mask = torch.zeros((max_obj,), dtype=torch.uint8).to("cuda")
            off2d = torch.zeros((max_obj, 2), dtype=torch.float32).to("cuda")
            wh = torch.zeros((max_obj, 2), dtype=torch.float32).to("cuda")
            hm = torch.zeros((hm_cls, output_h, output_w), dtype=torch.float32).to("cuda")
            draw_gaussian = draw_umich_gaussian
            for k in range(num_objs):
                h, w = gt_wh[k, 1], gt_wh[k, 0]
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                cls_id = static_label[k] - self.start_index
                draw_gaussian(hm[cls_id], ct_ints[k], radius)
                ind_[k] = ct_ints[k, 1] * output_w + ct_ints[k, 0]
                reg_mask[k] = 1
                off2d[k] = ct[k] - ct_ints[k]
                wh[k] = gt_wh[k]

            ind_ = ind_.detach()
            reg_mask = reg_mask.detach()
            hm = hm.detach()
            wh = wh.detach()
            off2d = off2d.detach()

            hm_loss = ct_focal_loss(barrier_hm, hm)

            barrier_off = tranpose_and_gather_feat(barrier_off, ind_)
            mask_off2d = reg_mask.detach()
            mask_off2d = mask_off2d.unsqueeze(1).expand_as(barrier_off).float()
            off2d_loss = F.l1_loss(barrier_off * mask_off2d, off2d * mask_off2d, size_average=False)
            off2d_loss = off2d_loss / (mask_off2d.sum() + 1e-4)

            barrier_wh = tranpose_and_gather_feat(barrier_wh, ind_)
            mask_wh = reg_mask.detach()
            mask_wh = mask_wh.unsqueeze(1).expand_as(barrier_wh).float()
            wh_loss = F.l1_loss(barrier_wh * mask_wh, wh * mask_wh, size_average=False)
            wh_loss = wh_loss * 0.1 / (mask_wh.sum() + 1e-4)

            hm_loss_list.append(hm_loss)
            off2d_loss_list.append(off2d_loss)
            wh_loss_list.append(wh_loss)

        # loss_dict['static_det_cls'] = torch.stack(hm_loss_list).mean(dim=0, keepdim=True)
        # loss_dict['static_det_off2d'] = torch.stack(off2d_loss_list).mean(dim=0, keepdim=True)
        # loss_dict['static_det_wh'] = torch.stack(wh_loss_list).mean(dim=0, keepdim=True)

        loss_dict["loss_static"] = torch.stack(hm_loss_list).mean(dim=0, keepdim=True) +  torch.stack(off2d_loss_list).mean(dim=0, keepdim=True) + torch.stack(wh_loss_list).mean(dim=0, keepdim=True)

        return loss_dict

    def _classes_loss(self, targets, classification):
        if targets is None:
            alpha_factor = torch.ones_like(classification) * self.alpha
            alpha_factor = alpha_factor.cuda()
            alpha_factor = 1. - alpha_factor
            focal_weight = classification
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
            bce = -(torch.log(1.0 - classification))
            loss = focal_weight * bce

        else:
            alpha_factor = torch.ones_like(targets) * self.alpha
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            loss = focal_weight * bce
            zeros = torch.zeros_like(loss)
            loss = torch.where(torch.ne(targets, -1.0), loss, zeros)
        return loss

    def _regress_loss(self, targets, regression):
        regression_diff = torch.abs(targets - regression)
        regression_loss = torch.where(
            torch.le(regression_diff, 1.0 / 9.0),
            0.5 * 9.0 * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / 9.0
        )
        return regression_loss.mean()
