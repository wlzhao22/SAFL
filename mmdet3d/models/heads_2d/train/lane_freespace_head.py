import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import kaiming_init
from mmdet3d.models.builder import HEADS
from abc import ABCMeta
from mmdet3d.models.utils.utils_2d.key_config import *
import numpy as np
from mmdet3d.models.utils.utils_2d.efficientdet_utils import tranpose_and_gather_feat
import math
from mmdet3d.models.utils.utils_2d.gaussian_target import draw_umich_gaussian, gaussian_radius
from mmdet3d.models.losses.losses_2d.efficientdet_loss import ct_focal_loss

BatchNorm2d = nn.SyncBatchNorm
bn_mom = 0.1


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

@HEADS.register_module()
class LaneFreespaceHead(nn.Module, metaclass=ABCMeta):
    def __init__(self,
                 min_distance=8,
                 radius=4,
                 lane_weight=4,
                 freespace_weight=1,
                 keypoint_class_weight=1,
                 keypoint_feature_weight=5,
                 down_ratio=4,
                 feature_dim=128,
                 layers=[2,2,2,2],
                 planes=96,
                 spp_planes=128,
                 head_planes=64,
                 with_freespace=True,
                 with_lane=True,
                 with_keypoint=True):
        super(LaneFreespaceHead, self).__init__()

        self.min_distance = min_distance
        self.radius=radius
        self.down_ratio = down_ratio
        highres_planes = planes * 2
        block = BasicBlock
        self.relu = nn.ReLU(inplace=False)
        self.tanh = nn.Tanh()
        self.with_keypoint = with_keypoint
        self.with_freespace = with_freespace
        self.with_lane = with_lane

        self.criteria_lane = torch.nn.BCEWithLogitsLoss().cuda()
        self.criteria_freespace = LabelSmoothing(smoothing=0.1)
        
        self.layer3 = self._make_layer(block, planes, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)


        self.compression3 = nn.Sequential(
                                        nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
                                        BatchNorm2d(highres_planes, momentum=bn_mom),
                                        )

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        self.down3 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   )

        self.down4 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 8, momentum=bn_mom),
                                   )

        self.layer3_ = self._make_layer(block, planes, highres_planes, 2)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)

        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        if self.with_lane:
            self.lane_head = self.build_head(inplanes=planes*4, interplanes=head_planes, outplanes=2, upsampling=8)
            self.lane_scSE = scSE(planes * 4)

        if self.with_freespace:
            self.freespace_head = self.build_head(inplanes=planes*4, interplanes=head_planes, outplanes=1)
            self.freespace_scSE = scSE(planes * 4)
        
        if self.with_keypoint:
            self.keypoint_hm_head = self.build_head(inplanes=planes*4, interplanes=head_planes, outplanes=1, upsampling=2)
            self.keypoint_feature_head = self.build_head(inplanes=planes*4, interplanes=head_planes, outplanes=feature_dim, upsampling=2)
            self.keypoint_off_head = self.build_head(inplanes=planes*4, interplanes=head_planes, outplanes=2, upsampling=2)
            self.keypoint_scSE = scSE(planes * 4)

        self.keypoint_feature_weight = keypoint_feature_weight 
        self.keypoint_class_weight = keypoint_class_weight
        self.lane_weight = lane_weight
        self.freespace_weight = freespace_weight
        
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)


    def build_head(self, inplanes, interplanes, outplanes, kernel_size=3, upsampling=1):
        bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        upsampling = nn.Upsample(scale_factor=upsampling,mode='bilinear') if upsampling > 1 else nn.Identity()
        return nn.Sequential(conv1,bn2,relu,conv2,upsampling)

    def forward_single(self, features, deploy=True):
        output = {}
        layers = []
        x = features[1]  # 1/8
        _, _, h, w = x.shape

        x_ = self.layer3_(self.relu(x))
        x = self.layer3(self.relu(x))
        layers.append(x)

        x = x + self.down3(self.relu(x_))
        x_ = x_ + F.interpolate(self.compression3(self.relu(layers[0])),size=[h, w],mode='bilinear')
        
        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(self.compression4(self.relu(layers[1])),size=[h, w], mode='bilinear')
        x_ = self.layer5_(self.relu(x_))

        x = F.interpolate(self.spp(self.layer5(self.relu(x))),size=[h, w],mode='bilinear')

        if self.with_lane:
            output[OUTPUT_LANE_SEG] = self.lane_head(self.lane_scSE(x + x_))

        if self.with_freespace:  
            output[OUTPUT_FREE_SPACE] = self.freespace_head(self.freespace_scSE(x + x_))
        
        if self.with_keypoint:
            keypoint_f = self.keypoint_scSE(x + x_)
            output[OUTPUT_KEYPOINT_HM] = self.keypoint_hm_head(keypoint_f)
            output[OUTPUT_KEYPOINT_OFF] = self.keypoint_off_head(keypoint_f)
            output[OUTPUT_KEYPOINT_FEATURE] = self.tanh(self.keypoint_feature_head(keypoint_f))
        
        if not deploy:
            return output
        else:
            if self.with_lane:
                output[OUTPUT_LANE_SEG] = output[OUTPUT_LANE_SEG].sigmoid()
    
            if self.with_freespace:
                output[OUTPUT_FREE_SPACE] = output[OUTPUT_FREE_SPACE].sigmoid()

            if self.with_keypoint:
                output[OUTPUT_KEYPOINT_HM] = output[OUTPUT_KEYPOINT_HM].sigmoid()
            
            return output

    def forward(self, features):
        return self.forward_single(features)

    def forward_train(self, inputs, outputs):
        outputs.update(self.forward_single(outputs[OUTPUT_BACKBONE_FEATURE_AFFINE], deploy=False))
        loss_dict = {}

        if self.with_lane:
            loss_dict.update(self.compute_lane_loss(outputs, inputs))

        if self.with_freespace:
            loss_dict.update(self.compute_freespace_loss(outputs, inputs))
        
        if self.with_keypoint:
            outputs_affine = self.forward_single(outputs[OUTPUT_FEATURE_AFFINE], deploy=False)
            loss_dict.update(self.compute_keypoint_loss(outputs, outputs_affine, inputs))
        return loss_dict

    def compute_lane_loss(self, outputs, inputs):
        loss_dict = {}
        gt_lane = inputs[COLLECTION_GT_LANE_SEG].squeeze(1).long()
        pred_lane = outputs[OUTPUT_LANE_SEG]
        gt_seg_mask = torch.zeros_like(gt_lane)
        gt_seg_mask[gt_lane>0] = 1
        gt_seg_mask = gt_seg_mask.unsqueeze(1)

        gt_seg_mask = torch.nn.functional.one_hot(gt_seg_mask, num_classes=2).squeeze()
        gt_seg_mask = gt_seg_mask.permute(0,3,1,2).float().contiguous()
        loss_dict["loss_lane_seg"] = self.lane_weight * self.criteria_lane(pred_lane, gt_seg_mask)
    
        return loss_dict

    def compute_freespace_loss(self, outputs, inputs):
        loss_dict = {}        
        gt_freespace = inputs[COLLECTION_GT_FREE_SPACE_RESIZED].squeeze(1).long()
        pred_freespace = outputs[OUTPUT_FREE_SPACE].squeeze(1)
        gt_cls = torch.argmax(gt_freespace,dim=1)
        loss_dict["loss_freespace_seg"] = self.freespace_weight * self.criteria_freespace(pred_freespace.permute(0,2,1), gt_cls)

        return loss_dict

    def compute_keypoint_loss(self, outputs, outputs_affine, inputs):
        gt_match_corner_pts, gt_corner_pts_affine, gt_corner_pts = inputs['match_corner_pts'], inputs['corner_pts_affine'], inputs['corner_pts']

        img_keypoint_hms, img_keypoint_features, img_keypoint_offs = outputs[OUTPUT_KEYPOINT_HM], outputs[OUTPUT_KEYPOINT_FEATURE], outputs[OUTPUT_KEYPOINT_OFF]
        img_affine_keypoint_hms, img_affine_keypoint_features, img_affine_keypoint_offs = outputs_affine[OUTPUT_KEYPOINT_HM], outputs_affine[OUTPUT_KEYPOINT_FEATURE], outputs_affine[OUTPUT_KEYPOINT_OFF]
        
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
            
            img_keypoint_hm = img_keypoint_hms[j]
            img_keypoint_feature = img_keypoint_features[j]
            img_keypoint_off = img_keypoint_offs[j]
            
            img_affine_keypoint_hm = img_affine_keypoint_hms[j]
            img_affine_keypoint_feature = img_affine_keypoint_features[j]
            img_affine_keypoint_off = img_affine_keypoint_offs[j]
            
            gt_match_corner_pt = gt_match_corner_pt / self.down_ratio
            gt_corner_pt = gt_corner_pt / self.down_ratio
            gt_corner_pt_affine = gt_corner_pt_affine / self.down_ratio

            gt_match_corner_pt_int = gt_match_corner_pt.to(torch.int)
            gt_corner_pt_int = gt_corner_pt.to(torch.int)
            gt_corner_pt_affine_int = gt_corner_pt_affine.to(torch.int)

            class_loss_1 = self._classes_loss(gt_corner_pt_int, gt_corner_pt, img_keypoint_hm, img_keypoint_off)
            class_loss_2 = self._classes_loss(gt_corner_pt_affine_int, gt_corner_pt_affine, img_affine_keypoint_hm, img_affine_keypoint_off)

            negative_feature_loss = self._feature_loss(gt_corner_pt_int, gt_corner_pt_affine_int, img_keypoint_feature, img_affine_keypoint_feature, False)    
            positive_feature_loss = self._feature_loss(gt_match_corner_pt_int[...,:2], gt_match_corner_pt_int[...,2:], img_keypoint_feature, img_affine_keypoint_feature, True)
            
            feature_loss = positive_feature_loss*2 - negative_feature_loss
            class_loss_list.extend([class_loss_1, class_loss_2])
            feature_loss_list.extend([feature_loss])

        loss_dict['keypoint_class_losses'] = self.keypoint_class_weight * torch.stack(class_loss_list).mean(dim=0, keepdim=True)
        loss_dict['keypoint_feature_losses'] = self.keypoint_feature_weight * torch.stack(feature_loss_list).mean(dim=0, keepdim=True)

        return loss_dict

    def _feature_loss(self, gt_corner_pt_int_1, gt_corner_pt_int_2, keypoint_feature_1, keypoint_feature_2, is_match):
        _, _, output_w = keypoint_feature_1.shape
        def gather_feature(gt_pt_int, hm_feature):
            ind = torch.zeros((1000,), dtype=torch.int64).cuda()
            track_mask = torch.zeros((1000,), dtype=torch.uint8).cuda()
            for i in range(gt_pt_int.shape[0]):
                ind[i] = gt_pt_int[i, 1] * output_w + gt_pt_int[i, 0]
                track_mask[i] = 1
            embedding = tranpose_and_gather_feat(hm_feature, ind)
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
            negative_embedding_1, negative_embedding_2 = get_nearest_negative_sample(embedding_1, embedding_2, self.min_distance)
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

    def compute_seg_loss(self, outputs, inputs):
        loss_dict = {}

        if self.with_lane:
            gt_lane = inputs[COLLECTION_GT_LANE_SEG].squeeze(1).long()
            pred_lane = outputs[OUTPUT_LANE_SEG]
            gt_seg_mask = torch.zeros_like(gt_lane)
            gt_seg_mask[gt_lane>0] = 1
            gt_seg_mask = gt_seg_mask.unsqueeze(1)

            gt_seg_mask = torch.nn.functional.one_hot(gt_seg_mask, num_classes=2).squeeze()
            gt_seg_mask = gt_seg_mask.permute(0,3,1,2).float().contiguous()
            loss_dict["loss_lane_seg"] = self.lane_weight * self.criteria_lane(pred_lane, gt_seg_mask)

        if self.with_freespace:
            gt_freespace = inputs[COLLECTION_GT_FREE_SPACE_RESIZED].squeeze(1).long()
            pred_freespace = outputs["freespace"].squeeze(1)
            gt_cls = torch.argmax(gt_freespace,dim=1)
            loss_dict["loss_freespace_seg"] = self.freespace_weight * self.criteria_freespace(pred_freespace.permute(0,2,1), gt_cls)

        return loss_dict
    

    def simple_test(self, features):
        pred = self.forward_single(features)
        return pred


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(2))
        nll_loss = nll_loss.squeeze(2)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm2d(branch_planes * 4, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 4, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):

        #x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear')+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear')+x_list[1]))))
        # x_list.append(self.process3((F.interpolate(self.scale3(x),
        #                 size=[height, width],
        #                 mode='bilinear')+x_list[2])))
        x_list.append(self.process3((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear')+x_list[2])))
       
        # out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        out_dappm = self.compression(torch.cat(x_list, 1))
        out = out_dappm + self.shortcut(x)

        return out



