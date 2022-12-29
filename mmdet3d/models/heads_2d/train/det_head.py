import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import itertools
from mmdet3d.models.utils.utils_2d.efficientdet_utils import tranpose_and_gather_feat, ConvBlock
from mmdet3d.models.utils.utils_2d.gaussian_target import draw_umich_gaussian, gaussian_radius
from mmdet3d.models.losses.losses_2d.efficientdet_loss import calc_iou, ct_focal_loss
from mmdet3d.models.utils.utils_2d.key_config import *
from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class DetHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, anchor_scale, \
                 ratios, scales, emb_dims=128, num_hm_cls=9, num_ids=1000, pyramid_levels=5, \
                 num_bbox_reg=4, num_ddd_reg=2, num_ddd_classes=13, onnx_export=False, with_tracking=False,
                 with_ddd=False, alpha=0.25, gamma=2.0):
        super(DetHead, self).__init__()

        self.regressor = Regressor(in_channels, num_anchors, num_layers, num_bbox_reg, pyramid_levels-1, onnx_export)
        self.classifier = Classifier(in_channels, num_anchors, num_classes, num_layers, pyramid_levels-1, onnx_export)
        self.anchor = Anchors(anchor_scale=anchor_scale, pyramid_levels=(torch.arange(pyramid_levels-1) + 3).tolist(),
                              ratios=ratios, scales=scales)

        self.hm = AnchorFreeModule(in_channels, num_hm_cls, onnx_export)
        self.wh = AnchorFreeModule(in_channels, 2, onnx_export)
        self.off = AnchorFreeModule(in_channels, 2, onnx_export)

        self.with_tracking = with_tracking
        if self.with_tracking:
            self.emb_scale = math.sqrt(2) * math.log(num_ids - 1)
            self.track = AnchorFreeModule(in_channels, emb_dims, onnx_export)
            self.track_classifier = nn.Linear(emb_dims, num_ids)

        self.with_ddd = with_ddd
        if self.with_ddd:
            self.ddd_regressor = Regressor(in_channels, num_anchors, num_layers, num_ddd_reg, pyramid_levels,
                                           onnx_export)
            self.ddd_classifier = Classifier(in_channels, num_anchors, num_ddd_classes, num_layers, pyramid_levels,
                                             onnx_export)

        self.alpha = alpha
        self.gamma = gamma

    def forward_single(self, features, img):
        outputs = {}
        features = features[-1]
        outputs[OUTPUT_DETECTION_DD_REGRESS] = self.regressor(features[1:])
        outputs[OUTPUT_DETECTION_DD_CLASSES] = self.classifier(features[1:])
        outputs[OUTPUT_DETECTION_DD_ANCHORS] = self.anchor(img, img.dtype)
        outputs[OUTPUT_DETECTION_OBJECT_HEATMAP] = self.hm(features[0])
        outputs[OUTPUT_DETECTION_OBJECT_WH] = self.wh(features[0])
        outputs[OUTPUT_DETECTION_OBJECT_OFF2D] = self.off(features[0])

        if self.with_tracking:
            outputs[OUTPUT_DETECTION_DD_EMBEDDINGS] = self.track(features[0])

        if self.with_ddd:
            outputs[OUTPUT_DETECTION_DDD_REGRESS] = self.ddd_regressor(features)
            outputs[OUTPUT_DETECTION_DDD_CLASSES] = self.ddd_classifier(features)

        return outputs

    def forward(self, features, img):
        return self.forward_single(features, img)

    def forward_train(self, inputs, outputs):
        outputs.update(self.forward_single(outputs[OUTPUT_FEATURE], inputs[COLLECTION_IMG]))
        detect_loss = self.loss(inputs, outputs)

        return detect_loss

    def loss(self, inputs, outputs):
        classifications, regressions, anchors = outputs[OUTPUT_DETECTION_DD_CLASSES], outputs[
            OUTPUT_DETECTION_DD_REGRESS], outputs[OUTPUT_DETECTION_DD_ANCHORS]
        gt_bboxes_2d, gt_labels = inputs[COLLECTION_GT_BBOX_2D], inputs[COLLECTION_GT_LABEL]
        barrier_hms, barrier_whs, barrier_offs = outputs[OUTPUT_DETECTION_OBJECT_HEATMAP], outputs[
            OUTPUT_DETECTION_OBJECT_WH], outputs[OUTPUT_DETECTION_OBJECT_OFF2D]
        barrier_hms = torch.clamp(barrier_hms.sigmoid_(), min=1e-4, max=1 - 1e-4)
        if self.with_ddd:
            ddd_classifications, ddd_regressions = outputs[OUTPUT_DETECTION_DDD_CLASSES], outputs[
                OUTPUT_DETECTION_DDD_REGRESS]
            gt_ddd_face_classes, gt_ddd_short_ys, gt_ddd_short_xs = inputs[COLLECTION_GT_DDD_FACE_CLASSES], inputs[
                COLLECTION_GT_DDD_SHORT_YS], inputs[COLLECTION_GT_DDD_SHORT_XS]

        if self.with_tracking:
            gt_reids = inputs[COLLECTION_GT_REID]
            embeddings = outputs[OUTPUT_DETECTION_DD_EMBEDDINGS]

        loss_dict = {}

        batch_size = classifications.shape[0]
        classification_losses = []
        ddd_classification_losses = []
        regression_losses = []
        tracking_losses = []
        ddd_regression_losses = []
        static_det_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):
            regression = regressions[j, :, :]
            classification = torch.clamp(classifications[j, :, :], 1e-4, 1.0 - 1e-4)
            gt_label = gt_labels[j]
            gt_bbox_2d = gt_bboxes_2d[j]

            dynamic_idx = gt_label < 6
            static_idx = gt_label >= 0

            dynamic_bbox_2d = gt_bbox_2d[dynamic_idx]
            dynamic_label = gt_label[dynamic_idx]

            static_bbox_2d = gt_bbox_2d[static_idx]
            static_label = gt_label[static_idx]

            dynamic_bbox_annotation = torch.cat([dynamic_bbox_2d, dynamic_label.unsqueeze(1).float()], dim=1)
            # static_bbox_annotation = torch.cat([static_bbox_2d, static_label.unsqueeze(1).float()],dim=1)

            gt_reid = gt_reids[j]
            gt_reid = gt_reid[dynamic_idx]

            if self.with_tracking:
                embedding = embeddings[j, :, :]
            if self.with_ddd:
                ddd_regression = ddd_regressions[j, :, :]
                ddd_classification = torch.clamp(ddd_classifications[j, :, :], 1e-4, 1.0 - 1e-4)
                dynamic_bbox_annotation = torch.cat(
                    [dynamic_bbox_annotation, gt_ddd_face_classes[j].unsqueeze(1).float(),
                     gt_ddd_short_ys[j].unsqueeze(1).float(), gt_ddd_short_xs[j].unsqueeze(1).float()], dim=1)

            if dynamic_bbox_annotation.shape[0] == 0:
                cls_loss = self._classes_loss(None, classification)
                regression_losses.append(torch.tensor(0).to(dtype).cuda())
                classification_losses.append(cls_loss.sum())
                if self.with_ddd:
                    ddd_loss = self._classes_loss(None, ddd_classification)
                    ddd_regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    ddd_classification_losses.append(ddd_loss.sum())

                if self.with_tracking:
                    tracking_losses.append(torch.tensor(0).to(dtype).cuda())

            else:

                IoU = calc_iou(anchor[:, :], dynamic_bbox_annotation[:, :4])

                IoU_max, IoU_argmax = torch.max(IoU, dim=1)

                # compute the loss for classification
                positive_indices = torch.ge(IoU_max, 0.5)
                num_positive_anchors = positive_indices.sum()
                assigned_annotations = dynamic_bbox_annotation[IoU_argmax, :]

                targets = torch.ones_like(classification) * -1
                targets[torch.lt(IoU_max, 0.4), :] = 0
                targets[positive_indices, :] = 0
                targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
                cls_loss = self._classes_loss(targets, classification)
                cls_loss = cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0)
                classification_losses.append(cls_loss)

                # ddd classification
                if self.with_ddd:
                    targets = torch.ones_like(ddd_classification) * -1
                    targets[torch.lt(IoU_max, 0.4), :] = 0
                    targets[positive_indices, :] = 0
                    targets[positive_indices, assigned_annotations[positive_indices, 5].long()] = 1
                    ddd_cls_loss = self._classes_loss(targets, ddd_classification)
                    ddd_cls_loss = ddd_cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0)
                    ddd_classification_losses.append(ddd_cls_loss)

                if positive_indices.sum() > 0:
                    assigned_annotations = assigned_annotations[positive_indices, :]

                    anchor_widths_pi = anchor_widths[positive_indices]
                    anchor_heights_pi = anchor_heights[positive_indices]
                    anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                    anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                    gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                    gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                    gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                    gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                    # efficientdet style
                    gt_widths = torch.clamp(gt_widths, min=1)
                    gt_heights = torch.clamp(gt_heights, min=1)

                    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                    targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                    targets_dw = torch.log(gt_widths / anchor_widths_pi)
                    targets_dh = torch.log(gt_heights / anchor_heights_pi)

                    targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                    targets = targets.t()
                    data = regression[positive_indices, :]
                    regression_loss = self._regress_loss(targets, regression[positive_indices, :])
                    regression_losses.append(regression_loss.mean())

                    # ddd reg
                    if self.with_ddd:
                        gt_short_y = assigned_annotations[:, 6]
                        gt_short_x = assigned_annotations[:, 7]
                        gt_short_y = torch.clamp(gt_short_y, min=1)
                        gt_short_x = torch.clamp(gt_short_x, min=1)
                        targets_short_y = torch.log(gt_short_y / anchor_heights_pi)
                        targets_short_x = torch.log(gt_short_x / anchor_widths_pi)
                        targets = torch.stack((targets_short_y, targets_short_x))
                        targets = targets.t()
                        ddd_regression_loss = self._regress_loss(targets, ddd_regression[positive_indices, :])
                        ddd_regression_losses.append(ddd_regression_loss.mean())

                    if self.with_tracking:
                        bbox_center_x = (dynamic_bbox_annotation[:, [0]] + dynamic_bbox_annotation[:, [2]]) // 16
                        bbox_center_y = (dynamic_bbox_annotation[:, [1]] + dynamic_bbox_annotation[:, [3]]) // 16
                        c, h, w = embedding.shape
                        ct_ints = torch.cat([bbox_center_x, bbox_center_y], dim=1)
                        ind = torch.zeros((100,), dtype=torch.int64).cuda()
                        track_mask = torch.zeros((100,), dtype=torch.uint8).cuda()
                        for i in range(ct_ints.shape[0]):
                            ind[i] = ct_ints[i, 1] * w + ct_ints[i, 0]
                            track_mask[i] = 1
                        embedding = self.tranpose_and_gather_feat(embedding, ind)
                        embedding = embedding[track_mask > 0].contiguous()
                        embedding = self.emb_scale * F.normalize(embedding)
                        re_ids = self.track_classifier(embedding).contiguous()
                        id_target = gt_reid
                        idloss = nn.CrossEntropyLoss(ignore_index=-1)
                        id_loss = idloss(re_ids, id_target.long())
                        tracking_losses.append(id_loss)
                else:
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    if self.with_ddd:
                        ddd_regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    if self.with_tracking:
                        tracking_losses.append(torch.tensor(0).to(dtype).cuda())

            # calculate loss for static objects
            barrier_hm = barrier_hms[j]
            barrier_wh = barrier_whs[j]
            barrier_off = barrier_offs[j]
            hm_cls, output_h, output_w = barrier_hm.shape
            down_ratio = 4
            feat_gt_boxes = static_bbox_2d / down_ratio
            feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                                   max=output_w - 1)
            feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                                   max=output_h - 1)
            feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                                feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])
            gt_wh = torch.stack([feat_ws, feat_hs], dim=1)
            ct = (torch.stack([(static_bbox_2d[:, 0] + static_bbox_2d[:, 2]) / 2,
                               (static_bbox_2d[:, 1] + static_bbox_2d[:, 3]) / 2],
                              dim=1) / down_ratio)

            ct_ints = (torch.stack([(static_bbox_2d[:, 0] + static_bbox_2d[:, 2]) / 2,
                                    (static_bbox_2d[:, 1] + static_bbox_2d[:, 3]) / 2],
                                   dim=1) / down_ratio).to(torch.int)
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
                cls_id = static_label[k]
                draw_gaussian(hm[cls_id], ct_ints[k], radius)
                ind_[k] = ct_ints[k, 1] * output_w + ct_ints[k, 0]
                reg_mask[k] = 1
                off2d[k] = ct[k] - ct_ints[k]
                wh[k] = gt_wh[k]

            # time1 = time.time()
            # img = hm.detach().permute(1,2,0).cpu().numpy()*255
            # for i in range(num_objs):
            #     cv2.rectangle(img,(feat_gt_boxes[i,0],feat_gt_boxes[i,1]),(feat_gt_boxes[i,2],feat_gt_boxes[i,3]),(0, 255, 0),1)
            # cv2.imwrite('/home/boden/Dev/ws/whz/BDPilot-merge/data/hm/%.5f.jpg'%time1,img)
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

            static_det_loss = hm_loss + off2d_loss + wh_loss
            static_det_losses.append(static_det_loss)

        loss_dict['classification_losses'] = torch.stack(classification_losses).mean(dim=0, keepdim=True)
        loss_dict['regression_losses'] = torch.stack(regression_losses).mean(dim=0, keepdim=True) * 50
        loss_dict['static_det_losses'] = torch.stack(static_det_losses).mean(dim=0, keepdim=True)

        if self.with_ddd:
            loss_dict['ddd_classification_losses'] = torch.stack(ddd_classification_losses).mean(dim=0, keepdim=True)
            loss_dict['ddd_regression_losses'] = torch.stack(ddd_regression_losses).mean(dim=0, keepdim=True) * 50

        if self.with_tracking:
            loss_dict['tracking_losses'] = torch.stack(tracking_losses).mean(dim=0, keepdim=True)

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

    def gather_feat(self, feat, ind, mask=None):
        dim = feat.size(1)
        ind = ind.unsqueeze(1).expand(ind.size(0), dim)
        feat = feat.gather(0, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(1, 2, 0).contiguous()  # 维度换位, continguous 内存地址变为连续 (1,152,272,512)
        feat = feat.view(-1, feat.size(2))  # 维度转换 (1,41334,512)
        feat = self.gather_feat(feat, ind)
        return feat


@HEADS.register_module()
class Regressor(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_layers, num_reg=4, pyramid_levels=5, onnx_export=False):
        super(Regressor, self).__init__()
        self.num_layers = num_layers
        self.num_reg = num_reg

        self.conv_list = nn.ModuleList(
            [ConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.header = ConvBlock(in_channels, num_anchors * self.num_reg, norm=False, activation=False)
        # self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        self.swish = nn.ReLU()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_reg)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)

        return feats


@HEADS.register_module()
class Classifier(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers, pyramid_levels=5, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [ConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.header = ConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        # self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        self.swish = nn.ReLU()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()

        return feats


class AnchorFreeModule(nn.Module):
    def __init__(self, in_channels, out_channels, onnx_export=False):
        super(AnchorFreeModule, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, inputs):
        feat = self.conv(inputs)

        return feat


class Anchors(nn.Module):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117
    """

    def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
        self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])

        self.last_anchors = {}
        self.last_shape = None

    def forward(self, image, dtype=torch.float32):
        """Generates multiscale anchor boxes.
        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.
        Returns:
          anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """
        image_shape = image.shape[2:]

        if image_shape == self.last_shape and image.device in self.last_anchors:
            return self.last_anchors[image.device]

        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape

        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)

        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)

        # save it for later use to reduce overhead
        self.last_anchors[image.device] = anchor_boxes
        return anchor_boxes