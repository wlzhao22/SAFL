import re
import itertools
import math
import collections
from functools import partial
import webcolors
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
import numpy as np
from torchvision.ops.boxes import nms as nms_torch
from torchvision.ops.boxes import batched_nms
from .utils_extra import Conv2dStaticSamePadding,MaxPool2dStaticSamePadding

def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)

def gather_feat(feat, ind, mask=None):
    if len(feat.shape)==2:
        dim = feat.size(1)
        ind = ind.unsqueeze(1).expand(ind.size(0), dim)
        feat = feat.gather(0, ind)
    else:
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)

    # if mask is not None:
    #     mask = mask.unsqueeze(2).expand_as(feat)
    #     feat = feat[mask]
    #     feat = feat.view(-1, dim)
    return feat

def tranpose_and_gather_feat(feat, ind):
    if len(feat.shape)==3:
        feat = feat.permute(1, 2, 0).contiguous()  # 维度换位, continguous 内存地址变为连续 (1,152,272,512)
        feat = feat.view(-1, feat.size(2))  # 维度转换 (1,41334,512)
    else:
        feat = feat.permute(0, 2, 3, 1).contiguous()  # 维度换位, continguous 内存地址变为连续 (1,152,272,512)
        feat = feat.view(feat.size(0), -1, feat.size(3))  # 维度转换 (1,41334,512)
    feat = gather_feat(feat, ind)
    return feat


def simple_nms(heat,kernel=3,out_heat=None):
    pad = (kernel -1)//2
    hmax = nn.functional.max_pool2d(heat,(kernel,kernel),stride=1,padding=pad)
    keep = (hmax==heat).float()
    out_heat = heat if out_heat is None else out_heat
    return out_heat * keep


def _topk(scores, topk):
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


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False):
        super(ConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.cov3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cov1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.active = nn.ReLU()

    def forward(self, x):
        x = self.cov3x3(x)
        x = self.cov1x1(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.active(x)

        return x


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



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


def regress_boxes(anchors, regression):
    y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
    x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]

    w = regression[..., 3].exp() * wa
    h = regression[..., 2].exp() * ha

    y_centers = regression[..., 0] * ha + y_centers_a
    x_centers = regression[..., 1] * wa + x_centers_a

    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    ymax = y_centers + h / 2.
    xmax = x_centers + w / 2.

    return torch.stack([xmin, ymin, xmax, ymax], dim=2)

def clip_boxes(boxes,h_max,w_max):

    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=w_max - 1)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=h_max - 1)

    return boxes


def anchor_based_post_process(anchors, regression, classification, embedding, threshold, iou_threshold, h_max, w_max,
                              scale):
    transformed_anchors = regress_boxes(anchors, regression)
    transformed_anchors = clip_boxes(transformed_anchors, h_max, w_max)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    print(embedding.shape)
    b, c, h, w = embedding.shape
    out = []
    for i in range(1):
        if scores_over_thresh[i].sum() == 0:
            bbox_result = np.array(())
            id_feature = np.array(())

            # out.append({
            #     'rois': np.array(()),
            #     'class_ids': np.array(()),
            #     'embeddings': np.array(()),
            #     'scores': np.array(()),
            # })
            continue
        # embedding_per = embedding[i, scores_over_thresh[i, :], ...]
        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = nms_torch(transformed_anchors_per, scores_per[:, 0], iou_threshold=iou_threshold)
        # anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)
        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            classes_ = classes_[:, np.newaxis]
            scores_ = scores_[anchors_nms_idx]
            scores_ = scores_[:, np.newaxis]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            bbox_center_x = (boxes_[:, 0] + boxes_[:, 2]) // 8
            bbox_center_y = (boxes_[:, 1] + boxes_[:, 3]) // 8
            pos_idx = bbox_center_y * w + bbox_center_x
            embeds_ = tranpose_and_gather_feat(embedding[i], pos_idx.long())

            boxes_ = boxes_ / scale

            bbox_result = np.concatenate((boxes_.cpu().numpy(), scores_.cpu().numpy(), classes_.cpu().numpy()), axis=1)

            id_feature = embeds_.cpu().numpy()
            # out.append({
            #     'rois': boxes_.cpu().numpy(),
            #     'class_ids': classes_.cpu().numpy(),
            #     'embeddings' embeds_.cpu().numpy(),
            #     'scores': scores_.cpu().numpy(),
            # })
        else:
            bbox_result = np.array(())
            id_feature = np.array(())
            # out.append({
            #     'rois': np.array(()),
            #     'class_ids': np.array(()),
            #     'embeddings': np.array(()),
            #     'scores': np.array(()),
            # })

    return bbox_result, id_feature

def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result

def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard

def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                    thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

def get_index_label(label, obj_list):
    index = int(obj_list.index(label))
    return index

STANDARD_COLORS = [
    'LawnGreen', 'Chartreuse', 'Aqua', 'Beige', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def regress_boxes_ddd(anchors, regression, ddd_regression):
    y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
    x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]

    w = regression[..., 3].exp() * wa
    h = regression[..., 2].exp() * ha

    y_centers = regression[..., 0] * ha + y_centers_a
    x_centers = regression[..., 1] * wa + x_centers_a

    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    ymax = y_centers + h / 2.
    xmax = x_centers + w / 2.

    ddd_regression_stack = None
    if ddd_regression is not None:
        short_y = ddd_regression[..., 0].exp() * ha
        short_x = ddd_regression[..., 1].exp() * wa
        ddd_regression_stack = torch.stack([short_y, short_x], dim=2)

    return torch.stack([xmin, ymin, xmax, ymax], dim=2), ddd_regression_stack

def post_process_ddd(anchors, regression, classification, embedding, threshold, iou_threshold, h_max, w_max, ddd_regression=None, ddd_classification=None):
    transformed_anchors, ddd_regression = regress_boxes_ddd(anchors, regression, ddd_regression)
    transformed_anchors = clip_boxes(transformed_anchors,h_max,w_max)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]


    bbox_result = np.array(())
    id_feature = np.array(())
    ddd_result = np.array(())
    out = []
    for i in range(1):
        content = {
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
                'ddd_clz_ids': np.array(()),
                'ddd_clz_scores': np.array(()),
                'ddd_regress': np.array(()),
                'id_feature': np.array(()),
            }
        if scores_over_thresh[i].sum() != 0:
            if embedding is not None:
                embedding_per = embedding[i, scores_over_thresh[i, :], ...]
            classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
            transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
            scores_per = scores[i, scores_over_thresh[i, :], ...]
            scores_, classes_ = classification_per.max(dim=0)
            anchors_nms_idx = nms_torch(transformed_anchors_per, scores_per[:, 0], iou_threshold=iou_threshold)

            if ddd_regression is not None:
                ddd_classification_per = ddd_classification[i, scores_over_thresh[i, :], :].permute(1, 0)
                ddd_scores_, ddd_classes_ = ddd_classification_per.max(dim=0)
                ddd_regression_per = ddd_regression[i, scores_over_thresh[i, :], ...]

            if anchors_nms_idx.shape[0] != 0:
                classes_ = classes_[anchors_nms_idx]
                classes_ = classes_[:,np.newaxis]
                scores_ = scores_[anchors_nms_idx]
                scores_ = scores_[:,np.newaxis]
                boxes_ = transformed_anchors_per[anchors_nms_idx, :]
                
                
                content.update({'rois': boxes_.cpu().numpy(),
                                'class_ids': classes_.cpu().numpy(),
                                'scores': scores_.cpu().numpy(), 
                                })

                if embedding is not None:
                    embeds_ = embedding_per[anchors_nms_idx,:]
                    content.update({'id_feature':embeds_.cpu().numpy()})

                if ddd_regression is not None:
                    ddd_classes_ = ddd_classes_[anchors_nms_idx]
                    ddd_scores_ = ddd_scores_[anchors_nms_idx]
                    ddd_regression_per = ddd_regression_per[anchors_nms_idx, :]
                    content.update({'ddd_clz_ids':ddd_classes_.cpu().numpy(),
                                    'ddd_clz_scores':ddd_scores_.cpu().numpy(),
                                    'ddd_regress':ddd_regression_per.cpu().numpy()})
            
        out.append(content)
    return out


def anchor_free_post_process(pred_heatmap, pred_wh, pred_off2d, scale, img_width, img_height):
    down_ratio = 4
    batch, cat, height, width = pred_heatmap.size()
    pred_heatmap = pred_heatmap.detach().sigmoid_()
    wh = pred_wh.detach()
    off2d = pred_off2d.detach()

    # perform nms on heatmaps
    heat = simple_nms(pred_heatmap)  # used maxpool to filter the max score

    topk = 100
    # (batch, topk)
    scores, inds, clses, ys, xs = _topk(heat, topk=topk)

    if pred_off2d is not None:
        off2d = tranpose_and_gather_feat(off2d, inds)
        off2d = off2d.view(batch, topk, 2)
        xs = (xs.view(batch, topk, 1) + off2d[:, :, 0:1]) * down_ratio
        ys = (ys.view(batch, topk, 1) + off2d[:, :, 1:2]) * down_ratio
    else:
        xs = xs.view(batch, topk, 1) * down_ratio
        ys = ys.view(batch, topk, 1) * down_ratio

    wh = tranpose_and_gather_feat(wh, inds)

    wh = wh.view(batch, topk, 2)
    wh = wh * down_ratio
    clses = clses.view(batch, topk, 1).float()
    scores = scores.view(batch, topk, 1)

    bboxes2d = torch.cat([xs - wh[..., [0]] / 2, ys - wh[..., [1]] / 2,
                          xs + wh[..., [0]] / 2, ys + wh[..., [1]] / 2], dim=2)

    result_list = []
    score_thr = 0.3
    for batch_i in range(bboxes2d.shape[0]):
        scores_per_img = scores[batch_i].squeeze()
        bboxes2d_per_img = bboxes2d[batch_i]
        labels_per_img = clses[batch_i]

        nms_idx = nms_torch(bboxes2d_per_img, scores_per_img, iou_threshold=0.5)

        bboxes2d_per_img = bboxes2d_per_img[nms_idx] / scale

        labels_per_img = labels_per_img[nms_idx]
        scores_per_img = scores_per_img[nms_idx]

        scores_keep = (scores_per_img > score_thr)

        scores_per_img = scores_per_img[scores_keep].unsqueeze(1)
        bboxes2d_per_img = bboxes2d_per_img[scores_keep]

        labels_per_img = labels_per_img[scores_keep]
        # bboxes2d_per_img[:, 0::2] = bboxes2d_per_img[:, 0::2].clamp(min=0, max=img_width - 1)
        # bboxes2d_per_img[:, 1::2] = bboxes2d_per_img[:, 1::2].clamp(min=0, max=img_height - 1)

        bboxes_per_img = torch.cat([bboxes2d_per_img, scores_per_img, labels_per_img], dim=1)
        # labels_per_img = labels_per_img.squeeze(-1)
        result_list.append(bboxes_per_img)
    result_list = torch.stack(result_list, dim=0)
    result_list = result_list.squeeze(0)
    result_list = result_list.cpu().numpy()
    return result_list
