import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from mmdet3d.models.utils.utils_2d.efficientdet_utils import tranpose_and_gather_feat, ConvBlock
from mmdet3d.models.utils.utils_2d.gaussian_target import draw_umich_gaussian, gaussian_radius
from mmdet3d.models.losses.losses_2d.efficientdet_loss import calc_iou, ct_focal_loss
from mmdet3d.models.utils.utils_2d.key_config import *
from mmdet3d.models.builder import HEADS


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse

class AnchorFreeModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AnchorFreeModule, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, U):
        f = self.conv(U)
        return f

@HEADS.register_module()
class KeyPointHead(nn.Module):
    def __init__(self, in_channels, feature_dim=256, down_ratio=4, num_hm_cls=1, alpha=0.25, gamma=2.0, radius=4, feature_weight=100, class_weight=1, c=8, m=1):
        super(KeyPointHead, self).__init__()
        self.hm = AnchorFreeModule(in_channels, 1)
        self.feature = AnchorFreeModule(in_channels, feature_dim)
        self.off = AnchorFreeModule(in_channels, 2)
        self.down_ratio = down_ratio
        self.alpha = alpha
        self.gamma = gamma
        self.radius = radius
        self.feature_weight = feature_weight 
        self.class_weight = class_weight
        self.tanh = nn.Tanh()
        self.c = c
        self.m = m
        self.scSE = scSE(in_channels)

    def forward_single(self, features):
        outputs = {}
        f = self.scSE(features[0])
        outputs["keypoint_hm"] = self.hm(f)
        outputs["keypoint_off"] = self.off(f)
        outputs["keypoint_feature_org"] = self.feature(f)
        outputs["keypoint_feature"] = self.tanh(outputs["keypoint_feature_org"])
        
        # pt_feature_norm = torch.norm(pt_feature, p=2, dim=1) # Compute the norm.
        # outputs["keypoint_feature"] = pt_feature.div(torch.unsqueeze(pt_feature_norm, 1)) # Divide by norm to normalize.
        # if self.training:
        #     outputs["keypoint_feature"] = self.tanh(outputs["keypoint_feature_org"])
        # else:
        #     # keypoint_feature = torch.zeros_like(outputs["keypoint_feature_org"])
        #     # keypoint_feature[outputs["keypoint_feature_org"]>=0] = 1
        #     # keypoint_feature[outputs["keypoint_feature_org"]<0] = -1
        #     # outputs["keypoint_feature"] = keypoint_feature
        #     outputs["keypoint_feature"] = self.tanh(outputs["keypoint_feature_org"])
        
        return outputs
    
    def forward(self, features):
        return self.forward_single(features)
    
    def forward_train(self, inputs, outputs):
        img_output = self.forward_single(outputs["img_backbone_feature"])
        img_affine_output = self.forward_single(outputs["img_affine_feature"])
        detect_loss = self.loss(inputs, outputs, img_output, img_affine_output)

        return detect_loss
    
    def loss(self, inputs, outputs, img_output, img_affine_output):
        gt_match_corner_pts, gt_corner_pts_affine, gt_corner_pts = inputs['match_corner_pts'], inputs['corner_pts_affine'], inputs['corner_pts']
        # gt_negative_corner_pts, gt_negative_affine_corner_pts = inputs['negative_corner_pts'], inputs['negative_affine_corner_pts']

        img_keypoint_hms, img_keypoint_features, img_keypoint_offs = img_output["keypoint_hm"], img_output["keypoint_feature"], img_output["keypoint_off"]
        img_affine_keypoint_hms, img_affine_keypoint_features, img_affine_keypoint_offs = img_affine_output["keypoint_hm"], img_affine_output["keypoint_feature"], img_affine_output["keypoint_off"]
        
        img_keypoint_hms = torch.clamp(img_keypoint_hms.sigmoid_(), min=1e-4, max=1 - 1e-4)
        img_affine_keypoint_hms = torch.clamp(img_affine_keypoint_hms.sigmoid_(), min=1e-4, max=1 - 1e-4)
       
        loss_dict = {}

        batch_size = img_keypoint_hms.shape[0]
        class_loss_list = []
        feature_loss_list = []
        for j in range(batch_size):
            gt_match_corner_pt = gt_match_corner_pts[j]
            gt_corner_pt = gt_corner_pts[j]
            gt_corner_pt_affine = gt_corner_pts_affine[j]
            # gt_negative_corner_pt = gt_negative_corner_pts[j]
            # gt_negative_affine_corner_pt = gt_negative_affine_corner_pts[j]
            
            img_keypoint_hm = img_keypoint_hms[j]
            img_keypoint_feature = img_keypoint_features[j]
            img_keypoint_off = img_keypoint_offs[j]
            
            img_affine_keypoint_hm = img_affine_keypoint_hms[j]
            img_affine_keypoint_feature = img_affine_keypoint_features[j]
            img_affine_keypoint_off = img_affine_keypoint_offs[j]
            
            gt_match_corner_pt = gt_match_corner_pt / self.down_ratio
            gt_corner_pt = gt_corner_pt / self.down_ratio
            gt_corner_pt_affine = gt_corner_pt_affine / self.down_ratio
            # gt_negative_corner_pt = gt_negative_corner_pt / self.down_ratio
            # gt_negative_affine_corner_pt = gt_negative_affine_corner_pt / self.down_ratio

            gt_match_corner_pt_int = gt_match_corner_pt.to(torch.int)
            gt_corner_pt_int = gt_corner_pt.to(torch.int)
            gt_corner_pt_affine_int = gt_corner_pt_affine.to(torch.int)
            # gt_negative_corner_pt_int = gt_negative_corner_pt.to(torch.int)
            # gt_negative_affine_corner_pt_int = gt_negative_affine_corner_pt.to(torch.int)

            class_loss_1 = self._classes_loss(gt_corner_pt_int, gt_corner_pt, img_keypoint_hm, img_keypoint_off)
            class_loss_2 = self._classes_loss(gt_corner_pt_affine_int, gt_corner_pt_affine, img_affine_keypoint_hm, img_affine_keypoint_off)

            # feature_loss_1 = -self.feature_weight * self._feature_loss(gt_corner_pt_int, gt_negative_corner_pt_int, img_keypoint_feature)
            # feature_loss_2 = -self.feature_weight * self._feature_loss(gt_corner_pt_affine_int, gt_negative_affine_corner_pt_int, img_affine_keypoint_feature)
            
            negative_feature_loss = self._feature_loss(gt_corner_pt_int, gt_corner_pt_affine_int, img_keypoint_feature, img_affine_keypoint_feature, False)    
            positive_feature_loss = self._feature_loss(gt_match_corner_pt_int[...,:2], gt_match_corner_pt_int[...,2:], img_keypoint_feature, img_affine_keypoint_feature, True)
            
            feature_loss = positive_feature_loss*2 - negative_feature_loss
            # feature_loss = torch.max(torch.Tensor(0).cuda(), positive_feature_loss*2 - negative_feature_loss + self.m*2)
            class_loss_list.extend([class_loss_1, class_loss_2])
            feature_loss_list.extend([feature_loss])

        loss_dict['keypoint_class_losses'] = self.class_weight * torch.stack(class_loss_list).mean(dim=0, keepdim=True)
        loss_dict['keypoint_feature_losses'] = self.feature_weight * torch.stack(feature_loss_list).mean(dim=0, keepdim=True)

        return loss_dict

    def _feature_loss(self, gt_corner_pt_int_1, gt_corner_pt_int_2, keypoint_feature_1, keypoint_feature_2, is_match):
        _, _, output_w = keypoint_feature_1.shape
        def gather_feature(gt_pt_int, hm_feature):
            ind = torch.zeros((1000,), dtype=torch.int64).cuda()
            track_mask = torch.zeros((1000,), dtype=torch.uint8).cuda()
            for i in range(gt_pt_int.shape[0]):
                ind[i] = gt_pt_int[i, 1] * output_w + gt_pt_int[i, 0]
                track_mask[i] = 1
            embedding = self.tranpose_and_gather_feat(hm_feature, ind)
            embedding = embedding[track_mask > 0].contiguous()
            return embedding

        def get_nearest_negative_sample(embedding_1, embedding_2, c):
            xy_list = []
            ind_1 = torch.zeros((embedding_1.shape[0],), dtype=torch.int64).cuda()
            ind_2 = torch.zeros((embedding_2.shape[0],), dtype=torch.int64).cuda()
            embedding_1_np = embedding_1.cpu().detach().numpy()
            embedding_2_np = embedding_2.cpu().detach().numpy()
            for id_1, pt_1 in enumerate(embedding_1_np):
                def distance(a, b):
                    return np.linalg.norm(a-b)
                
                distances = {distance(pt_1, pt_2): id_2 for id_2, pt_2 in enumerate(embedding_2_np)}
                match_key = None
                for key in sorted(distances.keys()): 
                    match_key = key
                    if key >= c:
                        break
                ind_1[id_1] = distances[match_key]
            
            for id_1, pt_1 in enumerate(embedding_2_np):
                def distance(a, b):
                    return np.linalg.norm(a-b)
                
                distances = {distance(pt_1, pt_2): id_2 for id_2, pt_2 in enumerate(embedding_1_np)}
                match_key = None
                for key in sorted(distances.keys()): 
                    match_key = key
                    if key >= c:
                        break
                ind_2[id_1] = distances[match_key]
            return embedding_2[ind_1], embedding_1[ind_2]

        embedding_1 = gather_feature(gt_corner_pt_int_1, keypoint_feature_1)
        embedding_2 = gather_feature(gt_corner_pt_int_2, keypoint_feature_2)

        if is_match:
            return F.mse_loss(embedding_1, embedding_2, reduction='mean')
        else:
            negative_embedding_1, negative_embedding_2 = get_nearest_negative_sample(embedding_1, embedding_2, self.c)
            return F.mse_loss(embedding_1, negative_embedding_1, reduction='mean') + F.mse_loss(embedding_2, negative_embedding_2, reduction='mean')
    def _classes_loss(self, gt_corner_pt_int, gt_corner_pt, keypoint_hm, keypoint_off):
        hm_cls, output_h, output_w = keypoint_hm.shape
        num_objs = gt_corner_pt_int.shape[0]
        max_obj = 1000  # 单张图片中最多多少个object
        ind_ = torch.zeros((max_obj,), dtype=torch.int64).to("cuda")
        reg_mask = torch.zeros((max_obj,), dtype=torch.uint8).to("cuda")
        off2d = torch.zeros((max_obj, 2), dtype=torch.float32).to("cuda")
        hm = torch.zeros((hm_cls, output_h, output_w), dtype=torch.float32).to("cuda")
        draw_gaussian = draw_umich_gaussian

        for k in range(num_objs):
            radius = gaussian_radius((math.ceil(self.radius), math.ceil(self.radius)))
            radius = max(0, int(radius))
            draw_gaussian(hm[hm_cls-1], gt_corner_pt_int[k], radius)
            ind_[k] = gt_corner_pt_int[k, 1] * output_w + gt_corner_pt_int[k, 0]
            reg_mask[k] = 1
            off2d[k] = gt_corner_pt[k] - gt_corner_pt_int[k]

        ind_ = ind_.detach()
        reg_mask = reg_mask.detach()
        hm = hm.detach()
        off2d = off2d.detach()

        hm_loss = ct_focal_loss(keypoint_hm, hm)

        keypoint_off = tranpose_and_gather_feat(keypoint_off, ind_)
        mask_off2d = reg_mask.detach()
        mask_off2d = mask_off2d.unsqueeze(1).expand_as(keypoint_off).float()
        off2d_loss = F.l1_loss(keypoint_off * mask_off2d, off2d * mask_off2d, size_average=False)
        off2d_loss = off2d_loss / (mask_off2d.sum() + 1e-4)
        
        return hm_loss + off2d_loss

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