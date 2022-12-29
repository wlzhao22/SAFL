import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


def gather_feat(feat, ind, mask=None):
    dim = feat.size(1)
    ind = ind.unsqueeze(1).expand(ind.size(0), dim)
    feat = feat.gather(0, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(1, 2, 0).contiguous()  # 维度换位, continguous 内存地址变为连续 (1,152,272,512)
    feat = feat.view(-1, feat.size(2))  # 维度转换 (1,41334,512)
    feat = gather_feat(feat, ind)
    return feat


def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


class DetLoss(nn.Module):
    def __init__(self):
        super(DetLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, embeddings, track_classifier, emb_scale, gt_bboxes_2d, gt_labels, gt_reids=None):

        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        tracking_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            embedding = embeddings[j,:,:]

            bbox_annotation = torch.cat([gt_bboxes_2d[j],gt_labels[j].unsqueeze(1).float(),gt_reids[j].unsqueeze(1).float()],dim=1)
            # bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():

                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = alpha_factor.cuda()
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce

                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    classification_losses.append(cls_loss.sum())
                else:

                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce

                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(cls_loss.sum())

                continue

            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4]) # anchor_num, ann_num

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)


            # compute the loss for classification
            targets = torch.ones_like(classification) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()
            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones_like(targets) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :] #Iou_max>=0.5的框

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

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
                
                #cal reid_loss
                # embedding = embedding[positive_indices,:]
                # embedding = emb_scale * F.normalize(embedding)
                # re_ids = track_classifier(embedding).contiguous()
                # idloss = nn.CrossEntropyLoss(ignore_index=-1)
                # id_loss = idloss(re_ids, assigned_annotations[:,5].long())
                # tracking_losses.append(id_loss)

                # tracking loss
                bbox_center_x = (bbox_annotation[:,[0]] + bbox_annotation[:,[2]])//8
                bbox_center_y = (bbox_annotation[:,[1]] + bbox_annotation[:,[3]])//8
                c,h,w = embedding.shape
                ct_ints = torch.cat([bbox_center_x,bbox_center_y],dim=1)
                ind = torch.zeros((100,),dtype=torch.int64).cuda()
                track_mask = torch.zeros((100,),dtype=torch.uint8).cuda()
                for i in range(ct_ints.shape[0]):
                    ind[i]= ct_ints[i,1] * w + ct_ints[i,0]
                    track_mask[i]=1           
                embedding = tranpose_and_gather_feat(embedding, ind)
                embedding = embedding[track_mask > 0].contiguous()
                embedding = emb_scale * F.normalize(embedding)
                re_ids = track_classifier(embedding).contiguous()
                id_target = bbox_annotation[:, 5]
                idloss = nn.CrossEntropyLoss(ignore_index=-1)
                id_loss = idloss(re_ids, id_target.long())
                tracking_losses.append(id_loss)

            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    tracking_losses.append(torch.tensor(0).to(dtype).cuda())

                else:
                    regression_losses.append(torch.tensor(0).to(dtype))
                    tracking_losses.append(torch.tensor(0).to(dtype))



        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True) * 50, \
               torch.stack(tracking_losses).mean(dim=0, keepdim=True),


def ct_focal_loss(pred, gt, gamma=2.0):
    """
    Focal loss used in CornerNet & CenterNet. Note that the values in gt (label) are in [0, 1] since
    gaussian is used to reduce the punishment and we treat [0, 1) as neg example.
    Args:
        pred: tensor, any shape.
        gt: tensor, same as pred.
        gamma: gamma in focal loss.
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)  # reduce punishment
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return neg_loss
    return (pos_loss + neg_loss) / num_pos