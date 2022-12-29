import torch
import torch.nn as nn
import numpy as numpy
import math
import numpy as np

def topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs

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

def gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()  # 维度换位, continguous 内存地址变为连续 (1,152,272,512)
    feat = feat.view(feat.size(0), -1, feat.size(3))  # 维度转换 (1,41334,512)
    feat = gather_feat(feat, ind)
    return feat

def get_alpha(rot):
    idx = (rot[:,:, 1:2] > rot[:,:, 5:6]).float()
    # alpha = torch.atan2(rot[:,:, 2:3], rot[:,:, 3:4])
    alpha1 = torch.atan2(rot[:,:, 2:3], rot[:,:, 3:4]) + (-0.5 * math.pi)
    alpha2 = torch.atan2(rot[:,:, 6:7], rot[:,:, 7:8]) + (0.5 * math.pi)


    return alpha1 * idx + alpha2 * (1 - idx)

def alpha2roty(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + torch.atan2(x - cx, fx)
    idx1 = (rot_y > math.pi).float()
    idx2 = (rot_y < -math.pi).float()
    rot_y -= 2 * math.pi * idx1
    rot_y += 2 * math.pi * idx2
    # if rot_y > math.pi:
    #     rot_y -= 2 * math.pi
    # if rot_y < -math.pi:
    #     rot_y += 2 * math.pi
    return rot_y   

def unproject_2d_to_3d(pt_2d, depth, P):
    # pt_2d: topk,2
    # depth: topk,1
    # P: batch, 3 x 4
    # return: topk,3
    z = depth - P[2, 3]
    z = depth - P[2, 3]
    x = (pt_2d[:,0:1] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
    y = (pt_2d[:,1:2] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
    pt_3d = torch.cat([x, y, z], dim=1)
    return pt_3d

def post_process(pred_heatmap, pred_wh, pred_off2d, img_metas, pred_wheel_off_2d=None, pred_hm_wheel=None,
                 pred_hm_wheel_off_2d=None, rescale=False, with_wheel=False):

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
        # xs = xs.view(batch, topk, 1) * self.down_ratio
        # ys = ys.view(batch, topk, 1) * self.down_ratio
        xs = (xs.view(batch, topk, 1) + off2d[:, :, 0:1]) * down_ratio
        ys = (ys.view(batch, topk, 1) + off2d[:, :, 1:2]) * down_ratio
    else:
        xs = xs.view(batch, topk, 1) * down_ratio
        ys = ys.view(batch, topk, 1) * down_ratio

    wh = tranpose_and_gather_feat(wh, inds)

    # wh = wh.view(batch, topk, 4)
    wh = wh.view(batch, topk, 2)
    wh = wh * down_ratio
    clses = clses.view(batch, topk, 1).float()
    scores = scores.view(batch, topk, 1)

    # bboxes2d = torch.cat([xs - wh[..., [0]], ys - wh[..., [1]],
    #                     xs + wh[..., [2]], ys + wh[..., [3]]], dim=2)

    bboxes2d = torch.cat([xs - wh[..., [0]] / 2, ys - wh[..., [1]] / 2,
                          xs + wh[..., [0]] / 2, ys + wh[..., [1]] / 2], dim=2)

    if with_wheel:
        dense_wheel_channel = wheel_channel = 4
        wheel_off_2d = pred_wheel_off_2d.detach()
        hm_wheel = pred_hm_wheel.detach().sigmoid_()
        hm_wheel_off_2d = pred_hm_wheel_off_2d.detach()
        wheel_off_2d = tranpose_and_gather_feat(wheel_off_2d, inds)
        wheel_off_2d = wheel_off_2d.view(batch, topk, -1)
        wheel_off_2d[:, :, 0::2] = xs + wheel_off_2d[:, :, 0::2] * down_ratio
        wheel_off_2d[:, :, 1::2] = ys + wheel_off_2d[:, :, 1::2] * down_ratio

        hm_wheel = simple_nms(hm_wheel)  # used maxpool to filter the max score
        wheel_off_2d = wheel_off_2d.view(batch, topk, dense_wheel_channel, 2).permute(0, 2, 1,
                                                                                      3).contiguous()  # b x J x K x 2
        reg_kps = wheel_off_2d.unsqueeze(3).expand(batch, dense_wheel_channel, topk, topk, 2)
        hm_score, hm_inds, hm_ys, hm_xs = topk_channel(hm_wheel, K=topk)  # b x J x K

        hm_wheel_off_2d = tranpose_and_gather_feat(hm_wheel_off_2d, hm_inds.view(batch, -1))
        hm_wheel_off_2d = hm_wheel_off_2d.view(batch, dense_wheel_channel, topk, 2)
        hm_xs = hm_xs + hm_wheel_off_2d[:, :, :, 0]
        hm_ys = hm_ys + hm_wheel_off_2d[:, :, :, 1]

        mask_score = 0.01
        mask = (hm_score > mask_score).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(2).expand(batch, dense_wheel_channel, topk, topk,
                                                                         2) * down_ratio
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, dense_wheel_channel, topk, 1, 1).expand(batch, dense_wheel_channel, topk, 1,
                                                                              2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, dense_wheel_channel, topk, 2)
        l = bboxes2d[:, :, 0].view(batch, 1, topk, 1).expand(batch, dense_wheel_channel, topk, 1)
        t = bboxes2d[:, :, 1].view(batch, 1, topk, 1).expand(batch, dense_wheel_channel, topk, 1)
        r = bboxes2d[:, :, 2].view(batch, 1, topk, 1).expand(batch, dense_wheel_channel, topk, 1)
        b = bboxes2d[:, :, 3].view(batch, 1, topk, 1).expand(batch, dense_wheel_channel, topk, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < mask_score) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, dense_wheel_channel, topk, 2)
        wheel = (1 - mask) * hm_kps + mask * wheel_off_2d
        wheel = wheel.permute(0, 2, 1, 3).contiguous().view(batch, topk, dense_wheel_channel * 2)
        hm_score = hm_score.permute(0, 2, 1, 3).contiguous().view(batch, topk, dense_wheel_channel)
        wheel = torch.cat((wheel, hm_score), dim=2)

    wheel_list = []
    result_list = []
    score_thr = 0.01
    for batch_i in range(bboxes2d.shape[0]):
        scores_per_img = scores[batch_i]
        scores_keep = (scores_per_img > score_thr).squeeze(-1)

        scores_per_img = scores_per_img[scores_keep]
        bboxes2d_per_img = bboxes2d[batch_i][scores_keep]

        labels_per_img = clses[batch_i][scores_keep]
        img_shape = img_metas[batch_i]['pad_shape']
        bboxes2d_per_img[:, 0::2] = bboxes2d_per_img[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
        bboxes2d_per_img[:, 1::2] = bboxes2d_per_img[:, 1::2].clamp(min=0, max=img_shape[0] - 1)

        if rescale:
            scale_factor = img_metas[batch_i]['scale_factor']
            bboxes2d_per_img /= bboxes2d_per_img.new_tensor(scale_factor)

        bboxes_per_img = torch.cat([bboxes2d_per_img, scores_per_img], dim=1)
        labels_per_img = labels_per_img.squeeze(-1)
        result_list.append((bboxes_per_img, labels_per_img))

        if with_wheel:
            wheel_per_img = wheel[batch_i][scores_keep]
            wheel_per_img[:, 0::2] = wheel_per_img[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
            wheel_per_img[:, 1::2] = wheel_per_img[:, 1::2].clamp(min=0, max=img_shape[0] - 1)
            if rescale:
                wheel_per_img /= wheel_per_img.new_tensor(np.array(scale_factor.tolist() * 2 + [1, 1, 1, 1]))
            wheel_list.append(wheel_per_img.cpu().numpy())

    if with_wheel:
        return result_list, wheel_list
    else:
        return result_list
# def post_process(pred_heatmap, pred_wh, pred_depth, pred_off2d, pred_off3d, pred_dims,
#                      pred_orien, pred_id, img_metas, calib, down_ratio=4,wh_agnostic=True,wh_gaussian=True,rescale=False):
#
#     batch, cat, height, width = pred_heatmap.size()
#     pred_heatmap = pred_heatmap.detach().sigmoid_()
#     wh = pred_wh.detach()
#     depth = 1. / (pred_depth.detach().sigmoid() + 1e-6) - 1.
#     id = pred_id.detach()
#     off2d = pred_off2d.detach()
#     off3d = pred_off3d.detach()
#     dims = pred_dims.detach()
#     orien = pred_orien.detach()
#
#
#     # perform nms on heatmaps
#     heat = simple_nms(pred_heatmap)  # used maxpool to filter the max score
#
#     topk = 100
#     # (batch, topk)
#     scores, inds, clses, ys, xs = _topk(heat, topk=topk)
#
#     if pred_off2d is not None:
#         off2d = tranpose_and_gather_feat(off2d,inds)
#         off2d = off2d.view(batch, topk, 2)
#         xs = (xs.view(batch, topk, 1) + off2d[:,:,0:1]) * down_ratio
#         ys = (ys.view(batch, topk, 1) + off2d[:,:,1:2]) * down_ratio
#     else:
#         xs = xs.view(batch, topk, 1) * down_ratio
#         ys = ys.view(batch, topk, 1) * down_ratio
#
#     id = tranpose_and_gather_feat(id, inds)
#     id = id.view(batch, topk, 128)
#
#     depth = tranpose_and_gather_feat(depth,inds)
#     depth = depth.view(batch,topk,1)
#
#     off3d = tranpose_and_gather_feat(off3d,inds)
#     off3d = off3d.view(batch,topk,2)
#
#     dims = tranpose_and_gather_feat(dims,inds)
#     dims = dims.view(batch,topk,3)
#
#     orien = tranpose_and_gather_feat(orien,inds)
#     orien = orien.view(batch,topk,8)
#     # alpha batch,topk,1
#     alpha = get_alpha(orien)
#
#
#     wh = tranpose_and_gather_feat(wh,inds)
#     if not wh_agnostic:
#         wh = wh.view(-1, topk, cat, 4)
#         wh = torch.gather(wh, 2, clses[..., None, None].expand(
#             clses.size(0), clses.size(1), 1, 4).long())
#
#     wh = wh.view(batch, topk, 4)
#     clses = clses.view(batch, topk, 1).float()
#     scores = scores.view(batch, topk, 1)
#
#     bboxes2d = torch.cat([xs - wh[..., [0]], ys - wh[..., [1]],
#                         xs + wh[..., [2]], ys + wh[..., [3]]], dim=2)
#
#     center_proj_2d = torch.cat([xs + off3d[..., [0]] * down_ratio, ys + off3d[..., [1]] * down_ratio],
#                                 dim=2)
#
#     result_list = []
#     id_feature = []
#     score_thr = 0.01
#     for batch_i in range(bboxes2d.shape[0]):
#         scores_per_img = scores[batch_i]
#         scores_keep = (scores_per_img > score_thr).squeeze(-1)
#
#         pred_id_per_img = id[batch_i][scores_keep]
#         scores_per_img = scores_per_img[scores_keep]
#         bboxes2d_per_img = bboxes2d[batch_i][scores_keep]
#         center_proj_2d_per_img = center_proj_2d[batch_i][scores_keep]
#         alpha_per_img = alpha[batch_i][scores_keep]
#         # rotation_y_per_img = rotation_y[batch_i][scores_keep]
#         depth_per_img = depth[batch_i][scores_keep]
#         dims_per_img = dims[batch_i][scores_keep]
#         # bboxes3d_per_img = bboxes3d[batch_i][scores_keep]
#         labels_per_img = clses[batch_i][scores_keep]
#         img_shape = img_metas[batch_i]['pad_shape']
#         bboxes2d_per_img[:, 0::2] = bboxes2d_per_img[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
#         bboxes2d_per_img[:, 1::2] = bboxes2d_per_img[:, 1::2].clamp(min=0, max=img_shape[0] - 1)
#
#         if rescale:
#             scale_factor = img_metas[batch_i]['scale_factor']
#             bboxes2d_per_img /= bboxes2d_per_img.new_tensor(scale_factor)
#             center_proj_2d_per_img /= torch.tensor(scale_factor[:2], device='cuda')
#
#         rotation_y_per_img = alpha2roty(alpha_per_img, center_proj_2d_per_img[:, [0]], calib[0, 2],calib[0, 0])
#         center_3d_per_img = unproject_2d_to_3d(center_proj_2d_per_img, depth_per_img, calib)
#         bboxes3d_per_img = torch.cat([center_3d_per_img, dims_per_img, rotation_y_per_img], dim=1)
#
#         bboxes_per_img = torch.cat([bboxes2d_per_img, bboxes3d_per_img, scores_per_img], dim=1)
#         labels_per_img = labels_per_img.squeeze(-1)
#         pred_id_per_img = pred_id_per_img.cpu().numpy()
#         result_list.append((bboxes_per_img, labels_per_img))
#         id_feature.append(pred_id_per_img)
#
#     return result_list, id_feature