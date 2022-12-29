import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init
import numpy as np
import math
from mmdet.core import multi_apply
from mmdet.core.anchor import calc_region
from mmdet3d.models.losses.losses_2d.focal_loss import ct_focal_loss
from mmdet3d.models.losses.losses_2d.iou_loss import ttf_giou_loss
from mmdet3d.models.losses.losses_2d import BinRotLoss
from mmdet.models.builder import HEADS
from abc import ABCMeta
from mmdet3d.models.utils.utils_2d.key_config import *



@HEADS.register_module()
class Detect3DHead(nn.Module,metaclass=ABCMeta):

    def __init__(self,
                 n_id = 5983,
                 id_dim = 128,
                 num_classes=81,
                 wh_offset_base=16.,
                 wh_area_process='log',
                 wh_agnostic=True,
                 wh_gaussian=True,
                 alpha=0.54,
                 beta=0.54,
                 hm_weight=1.,
                 wh_weight=1.,
                 max_objs=128,
                 train_cfg=None,
                 test_cfg=None,
                 fix_para=False,):
        super(Detect3DHead, self).__init__()
        assert wh_area_process in [None, 'norm', 'log', 'sqrt']

        self.num_classes = num_classes
        self.wh_offset_base = wh_offset_base
        self.wh_area_process = wh_area_process
        self.wh_agnostic = wh_agnostic
        self.wh_gaussian = wh_gaussian
        self.alpha = alpha
        self.beta = beta
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.max_objs = max_objs
        self.fp16_enabled = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.id_dim = id_dim
        self.n_id = n_id

        self.down_ratio = 4
        self.num_fg = num_classes
        self.wh_planes = 4
        self.base_loc = None
        self.emb_scale = math.sqrt(2) * math.log(self.n_id - 1)

        self.off2d_channel = 2
        self.off3d_channel = 2
        self.orien_channel = 8
        self.dims_channel = 3
        self.depth_channel = 1


        self.first_level = int(np.log2(4))
        channels = [16, 32, 64, 128, 256, 512]
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)
        # heads
        self.heads_channels = dict(hm=self.num_fg, wh=self.wh_planes, off_2d=self.off2d_channel,
                                       off_3d=self.off3d_channel, dims=self.dims_channel, orien=self.orien_channel,
                                       depth=self.depth_channel, id=self.id_dim)

        for head in self.heads_channels:
            num_output = self.heads_channels[head]
            fc = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=(3, 1), padding=(1, 0), bias=True),
                nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1), bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, num_output, kernel_size=1, stride=1, padding=0))
            self.__setattr__('{}'.format(head), fc)
        self.classifier = nn.Linear(self.id_dim, self.n_id)
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward_single(self, feats):
        """
        Args:
            feats: list(tensor).
        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        x = self.dla_up(feats[self.first_level:])
        ret = {}
        for head in self.heads_channels:
            fpn_out = self.__getattr__('{}'.format(head))(x)
            ret[head] = fpn_out

        return ret

    def forward(self, feats):

        return self.forward_single(feats)


    def project_3d_to_2d(self, pts_3d, P):
        # pts_3d: n x 3
        # P: 3 x 4
        # return: n x 2
        pts_3d_homo = torch.cat([pts_3d, torch.ones((pts_3d.size()[0], 1), dtype=torch.float32,device='cuda')], dim=1)
        pts_2d = torch.matmul(P.float(), pts_3d_homo.permute(1, 0)).permute(1, 0)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
        return pts_2d


    def gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous() 
        feat = feat.view(feat.size(0), -1, feat.size(3))  
        feat = self.gather_feat(feat, ind)
        return feat

    def forward_train(self, inputs, outputs):
        feature, gt_bboxes_2d, gt_bboxes_3d, gt_labels, gt_reids, img_metas, calib = \
            outputs[OUTPUT_FEATURE_AUG], inputs[COLLECTION_GT_BBOX_2D], inputs[COLLECTION_GT_BBOX_3D], \
                inputs[COLLECTION_GT_LABEL], inputs[COLLECTION_GT_REID], inputs[COLLECTION_IMG_METAS], \
                    inputs[COLLECTION_CALIB]
        pred = self.forward_single(feature)
        detect_loss = self.loss(pred_heatmap=pred['hm'], pred_wh=pred['wh'], pred_depth=pred['depth'],
                                pred_off2d=pred['off_2d'], pred_off3d=pred['off_3d'], pred_dims=pred['dims'],
                                pred_orien=pred['orien'],
                                pred_id=pred['id'], gt_bboxes_2d=gt_bboxes_2d, gt_bboxes_3d=gt_bboxes_3d,
                                gt_labels=gt_labels, gt_reids=gt_reids,
                                img_metas=img_metas, calib=calib)


        return detect_loss


    def loss(self, pred_heatmap, pred_wh, pred_depth, pred_off2d, pred_off3d, pred_dims, pred_orien,
             pred_id, gt_bboxes_2d, gt_bboxes_3d, gt_labels, gt_reids, img_metas, calib):

        all_targets = self.target_generator(gt_bboxes_2d, gt_bboxes_3d, gt_labels, gt_reids, img_metas, calib)
        detection_2d_loss, detection_3d_loss, tracking_loss = self.loss_calc(
            pred_heatmap, pred_wh, pred_depth, pred_off2d, pred_off3d, pred_dims, pred_orien,
            pred_id, *all_targets)
        return {'losses/loss_detection_2d': detection_2d_loss, 'losses/loss_detection_3d': detection_3d_loss,
                'losses/loss_tracking':tracking_loss}


    def _topk(self, scores, topk):
        batch, cat, height, width = scores.size()

        # both are (batch, 80, topk)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), topk)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # both are (batch, topk). select topk from 80*topk
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
        topk_clses = (topk_ind / topk).int()
        topk_ind = topk_ind.unsqueeze(2)
        topk_inds = topk_inds.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_ys = topk_ys.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_xs = topk_xs.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = heatmap.new_tensor(gaussian)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                          w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap


    def bbox_areas(self, bboxes, keep_axis=False):
        x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (y_max - y_min + 1) * (x_max - x_min + 1)
        if keep_axis:
            return areas[:, None]
        return areas


    def target_single_image(self, gt_bboxes_2d, gt_bboxes_3d, gt_labels, gt_ids, img_metas, calib, feat_shape):

        calib = calib.cuda()
        output_h, output_w = feat_shape
        heatmap_channel = self.num_fg

        scale_factor = img_metas['scale_factor'][0:2]
        scale_factor = torch.tensor(scale_factor,device='cuda')

        heatmap = gt_bboxes_2d.new_zeros((heatmap_channel, output_h, output_w))
        fake_heatmap = gt_bboxes_2d.new_zeros((output_h, output_w))
        box_target = gt_bboxes_2d.new_ones((self.wh_planes, output_h, output_w)) * -1
        reg_weight = gt_bboxes_2d.new_zeros((self.wh_planes // 4, output_h, output_w))

        if self.wh_area_process == 'log':
            boxes_areas_log = self.bbox_areas(gt_bboxes_2d).log()
        elif self.wh_area_process == 'sqrt':
            boxes_areas_log = self.bbox_areas(gt_bboxes_2d).sqrt()
        else:
            boxes_areas_log = self.bbox_areas(gt_bboxes_2d)
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))

        if self.wh_area_process == 'norm':
            boxes_area_topk_log[:] = 1.

        gt_boxes = gt_bboxes_2d[boxes_ind]
        gt_labels = gt_labels[boxes_ind]
        gt_ids = gt_ids[boxes_ind]
        gt_center_3d = gt_bboxes_3d[:,0:3][boxes_ind]
        gt_dims = gt_bboxes_3d[:,3:6][boxes_ind]
        gt_alpha = gt_bboxes_3d[:,6][boxes_ind]


        feat_gt_boxes = gt_boxes / self.down_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                               max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                               max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio)

        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        proj_center3d = self.project_3d_to_2d(gt_center_3d, calib)
        proj_center3d = (proj_center3d * scale_factor) / self.down_ratio
        # ind = (ct_ints[:,1]*feat_ws + ct_ints[:,0]).to(torch.int)
        num_objs = gt_labels.shape[0]
        max_obj = 100 #单张图片中最多多少个objuect
        ind = torch.zeros((max_obj,),dtype=torch.int64).to("cuda")
        reg_mask = torch.zeros((max_obj,),dtype=torch.uint8).to("cuda")
        # rot_mask = torch.zeros((max_obj,), dtype=torch.uint8).to("cuda")
        ids = torch.zeros((max_obj,),dtype=torch.int64).to("cuda")
        depth = torch.zeros((max_obj,1),dtype=torch.int64).to("cuda")
        rotbin = torch.zeros((max_obj, 2), dtype=torch.int64).to("cuda")
        rotres = torch.zeros((max_obj, 2), dtype=torch.float32).to("cuda")
        off2d = torch.zeros((max_obj, 2), dtype=torch.float32).to("cuda")
        off3d = torch.zeros((max_obj, 2), dtype=torch.float32).to("cuda")
        dim = torch.zeros((max_obj, 3), dtype=torch.float32).to("cuda")
        for k in range(num_objs):
            ind[k] = ct_ints[k,1] * output_w + ct_ints[k,0]
            reg_mask[k] = 1
            ids[k] = gt_ids[k]
            depth[k]= gt_center_3d[k][2]
            off2d[k] = ct[k] - ct_ints[k]
            off3d[k] = proj_center3d[k] - ct[k]
            dim[k] = gt_dims[k]
            if gt_alpha[k] < math.pi / 6. or gt_alpha[k] > 5 * math.pi / 6.:
                rotbin[k, 0] = 1
                rotres[k, 0] = gt_alpha[k] - (-0.5 * math.pi)
            if gt_alpha[k] > -math.pi / 6. or gt_alpha[k] < -5 * math.pi / 6.:
                rotbin[k, 1] = 1
                rotres[k, 1] = gt_alpha[k] - (0.5 * math.pi)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
        if self.wh_gaussian and self.alpha != self.beta:
            h_radiuses_beta = (feat_hs / 2. * self.beta).int()
            w_radiuses_beta = (feat_ws / 2. * self.beta).int()

        if not self.wh_gaussian:
            # calculate positive (center) regions
            r1 = (1 - self.beta) / 2
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = calc_region(gt_boxes.transpose(0, 1), r1)
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = [torch.round(x.float() / self.down_ratio).int()
                                                  for x in [ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s]]
            ctr_x1s, ctr_x2s = [torch.clamp(x, max=output_w - 1) for x in [ctr_x1s, ctr_x2s]]
            ctr_y1s, ctr_y2s = [torch.clamp(y, max=output_h - 1) for y in [ctr_y1s, ctr_y2s]]

        # larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k]

            fake_heatmap = fake_heatmap.zero_()
            self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item(), w_radiuses_alpha[k].item())
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)

            if self.wh_gaussian:
                if self.alpha != self.beta:
                    fake_heatmap = fake_heatmap.zero_()
                    self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                                h_radiuses_beta[k].item(),
                                                w_radiuses_beta[k].item())
                box_target_inds = fake_heatmap > 0
            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[k], ctr_y1s[k], ctr_x2s[k], ctr_y2s[k]
                box_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1

            if self.wh_agnostic:
                box_target[:, box_target_inds] = gt_boxes[k][:, None]
                cls_id = 0
            else:
                box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = gt_boxes[k][:, None]

            if self.wh_gaussian:
                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
            else:
                reg_weight[cls_id, box_target_inds] = \
                    boxes_area_topk_log[k] / box_target_inds.sum().float()

        return heatmap, box_target, reg_weight, ind, reg_mask, ids, depth, off2d, off3d, dim, rotbin, rotres


    def target_generator(self, gt_bboxes_2d, gt_bboxes_3d, gt_labels, gt_ids, img_metas, calib):
        with torch.no_grad():
            feat_shape = (img_metas[0]['pad_shape'][0] // self.down_ratio,
                          img_metas[0]['pad_shape'][1] // self.down_ratio)
            heatmap, box_target, reg_weight, ind, reg_mask,ids, depth, off2d, off3d, dim, rotbin, rotres = multi_apply(
                self.target_single_image,
                gt_bboxes_2d,
                gt_bboxes_3d,
                gt_labels,
                gt_ids,
                img_metas,
                calib,
                feat_shape=feat_shape
            )

            heatmap, box_target = [torch.stack(t, dim=0).detach() for t in [heatmap, box_target]]
            reg_weight = torch.stack(reg_weight, dim=0).detach()
            ind = torch.stack(ind, dim=0).detach()
            reg_mask = torch.stack(reg_mask, dim=0).detach()
            ids = torch.stack(ids,dim=0).detach()
            depth = torch.stack(depth, dim=0).detach()
            off2d = torch.stack(off2d, dim=0).detach()
            off3d = torch.stack(off3d, dim=0).detach()
            dim = torch.stack(dim, dim=0).detach()
            rotbin = torch.stack(rotbin, dim=0).detach()
            rotres = torch.stack(rotres, dim=0).detach()

            return heatmap, box_target, reg_weight, ind, reg_mask, ids, depth, off2d, off3d, dim, rotbin, rotres

    def loss_calc(self, pred_hm, pred_wh, pred_depth, pred_off2d, pred_off3d, pred_dims, pred_orien,
                  pred_id, heatmap, box_target, wh_weight, ind, reg_mask,
                  ids, depth, off2d, off3d, dims, rotbin, rotres):
        """
        Args:
            pred_hm: tensor, (batch, 80, h, w).
            pred_wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            heatmap: tensor, same as pred_hm.
            pred_id: tensor, (batch,nid_dim). (one-hot)
            box_target: tensor, same as pred_wh.
            gt_id, tensor, same as pred_id. (one-hot)
            wh_weight: tensor, same as pred_wh.
        Returns:
            hm_loss
            wh_loss
        """
        H, W = pred_hm.shape[2:]
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        hm_loss = ct_focal_loss(pred_hm, heatmap) * self.hm_weight

        pred_depth = 1. / (pred_depth.sigmoid() + 1e-6) - 1.
        pred_depth = self.tranpose_and_gather_feat(pred_depth,ind)
        mask_depth = reg_mask.detach()
        mask_depth = mask_depth.unsqueeze(2).expand_as(pred_depth).float()
        # mask_depth = reg_mask.unsqueeze(2).expand_as(pred_depth).float()
        depth_loss = F.l1_loss(pred_depth * mask_depth, depth * mask_depth, reduction='elementwise_mean')

        pred_dims = self.tranpose_and_gather_feat(pred_dims,ind)
        mask_dims = reg_mask.detach()
        mask_dims = mask_dims.unsqueeze(2).expand_as(pred_dims).float()
        # mask_dims = reg_mask.unsqueeze(2).expand_as(pred_dims).float()
        dims_loss = F.l1_loss(pred_dims * mask_dims, dims * mask_dims, reduction='elementwise_mean')

        pred_off2d = self.tranpose_and_gather_feat(pred_off2d,ind)
        mask_off2d = reg_mask.detach()
        mask_off2d = mask_off2d.unsqueeze(2).expand_as(pred_off2d).float()
        # mask_off2d = reg_mask.unsqueeze(2).expand_as(pred_off2d).float()
        off2d_loss = F.l1_loss(pred_off2d * mask_off2d, off2d * mask_off2d, reduction='elementwise_mean')

        pred_off3d = self.tranpose_and_gather_feat(pred_off3d,ind)
        mask_off3d = reg_mask.detach()
        mask_off3d = mask_off3d.unsqueeze(2).expand_as(pred_off3d).float()
        # mask_off3d = reg_mask.unsqueeze(2).expand_as(pred_off3d).float()
        off3d_loss = F.l1_loss(pred_off3d * mask_off3d, off3d * mask_off3d, reduction='elementwise_mean')

        pred_orien = self.tranpose_and_gather_feat(pred_orien,ind)
        # mask_orien = reg_mask.detach()
        # mask_orien = mask_orien.unsqueeze(2).expand_as(pred_orien).float()
        # mask_orien = reg_mask.unsqueeze(2).expand_as(pred_orien).float()
        binrotloss = BinRotLoss()
        orien_loss = binrotloss(pred_orien,rotbin,rotres,reg_mask)


        mask = wh_weight.view(-1, H, W)
        avg_factor = mask.sum() + 1e-4

        if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
            base_step = self.down_ratio
            shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        # (batch, h, w, 4)
        pred_boxes = torch.cat((self.base_loc - pred_wh[:, [0, 1]],
                                self.base_loc + pred_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
        # (batch, h, w, 4)
        boxes = box_target.permute(0, 2, 3, 1)
        wh_loss = ttf_giou_loss(pred_boxes, boxes, mask, avg_factor=avg_factor) * self.wh_weight

        pred_id = self.tranpose_and_gather_feat(pred_id,ind)
        pred_id = pred_id[reg_mask>0].contiguous()
        pred_id = self.emb_scale * F.normalize(pred_id)
        id_target = ids[reg_mask>0]
        id_output = self.classifier(pred_id).contiguous()
        idloss = nn.CrossEntropyLoss(ignore_index=-1)
        id_loss = idloss(id_output, id_target)

        detection_2d_loss = hm_loss + wh_loss + off2d_loss
        detection_3d_loss = depth_loss + dims_loss + off3d_loss + orien_loss
        tracking_loss = id_loss


        return detection_2d_loss,detection_3d_loss,tracking_loss

BatchNorm = nn.BatchNorm2d

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class DLAUp(nn.Module):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(3, channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])
            layers[-i - 1:] = y
        return x

class IDAUp(nn.Module):
    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(
                    nn.Conv2d(c, out_dim,
                              kernel_size=1, stride=1, bias=False),
                    BatchNorm(out_dim),
                    nn.ReLU(inplace=True))
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(
                    out_dim, out_dim, f * 2, stride=f, padding=f // 2,
                    output_padding=0, groups=out_dim, bias=False)
                fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim,
                          kernel_size=node_kernel, stride=1,
                          padding=node_kernel // 2, bias=False),
                BatchNorm(out_dim),
                nn.ReLU(inplace=True))
            setattr(self, 'node_' + str(i), node)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
        return x, y
