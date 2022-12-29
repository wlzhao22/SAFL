import torch 
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import kaiming_init
from mmdet3d.models.builder import HEADS
from abc import ABCMeta
from mmdet3d.models.utils.utils_2d.key_config import *
from mmdet3d.models.losses.losses_2d.efficientdet_loss import ct_focal_loss
from mmdet3d.models.losses.losses_2d.focal_loss_seg import FocalLoss2d
from mmdet3d.models.utils.utils_2d.gaussian_target import draw_umich_gaussian, gaussian_radius
BatchNorm2d = nn.SyncBatchNorm


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

class SegmentationModule(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, upsampling=1, stride=1):
        super(SegmentationModule, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False, stride=stride),
            nn.BatchNorm2d(interplanes),
            nn.ReLU(inplace=False),
            nn.Conv2d(interplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Upsample(scale_factor=upsampling,mode='bilinear') if upsampling > 1 else nn.Identity(),
            # nn.Upsample(size=(96, 192),mode='bilinear'),
        )
    
    def forward(self, inputs):
        return self.conv(inputs)

class CenterModule(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, upsampling=1, stride=2):
        super(CenterModule, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False, stride=stride),
            nn.BatchNorm2d(interplanes),
            nn.ReLU(inplace=False),
            nn.Conv2d(interplanes, interplanes, kernel_size=3, padding=1, bias=False, stride=1),
            nn.BatchNorm2d(interplanes),
            nn.ReLU(inplace=False),
            nn.Conv2d(interplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Upsample(scale_factor=upsampling,mode='bilinear') if upsampling > 1 else nn.Identity(),
            # nn.Upsample(size=(96, 192),mode='bilinear'),
        )
    
    def forward(self, inputs):
        return self.conv(inputs)

@HEADS.register_module()
class LaneHeadStu(nn.Module, metaclass=ABCMeta):
    def __init__(self,
                 weight=1,
                 lane_range = [1, 2, 3, 4],
                #  seg_size=[96, 192],
                 seg_size=[48, 96],
                 heat_size=[24, 48],
                 img_size=[384, 768],
                 in_channels=[64,64],):
        super(LaneHeadStu, self).__init__()

        # self.cSE = cSE(in_channels[1])
        # self.block = BasicBlock(in_channels[1], in_channels[1])
        self.weight = weight
        self.lane_range = lane_range

        self.seg_size = seg_size
        self.heat_size = heat_size
        self.img_size = img_size

        self.exist_head = CenterModule(inplanes=in_channels[1], interplanes=in_channels[1]//2, outplanes=1)
        self.segmentation_head_x = SegmentationModule(inplanes=in_channels[0], interplanes=in_channels[0]//2, outplanes=self.heat_size[1]+1)
        self.segmentation_head_y = SegmentationModule(inplanes=in_channels[0], interplanes=in_channels[0]//2, outplanes=self.heat_size[0]+1)
        self.regression_head = SegmentationModule(inplanes=in_channels[0], interplanes=in_channels[0]//2, outplanes=2)
        self.x_weights = torch.ones((self.heat_size[1]+1)).cuda()
        self.x_weights[0] = 0.2
        self.y_weights = torch.ones((self.heat_size[0]+1)).cuda()
        self.y_weights[0] = 0.2

        self.focal_loss_x = FocalLoss2d()
        self.focal_loss_y = FocalLoss2d()

        self.adaptations_nk= nn.ModuleList([
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)])

        self.adaptations_bk= nn.ModuleList([
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 1280, kernel_size=1, stride=1, padding=0)])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward_single(self, features):
        output = {}
        f = features[0]
        output[OUTPUT_LANE_EXIST] = self.exist_head(f)
        output[OUTPUT_LANE_SEG_X] = self.segmentation_head_x(f)
        output[OUTPUT_LANE_SEG_Y] = self.segmentation_head_y(f)
        output[OUTPUT_LANE_REG_XY] = self.regression_head(f)
   
        return output

    def forward(self, features):
        return self.forward_single(features)

    def forward_train(self, inputs, outputs):
        outputs.update(self.forward_single(outputs[OUTPUT_STUDENT_FEATURE_AUG]))
        return self.compute_loss(inputs, outputs)

    def compute_loss(self, inputs, outputs):
        loss_dict = {}

        hint_loss_bk, hint_loss_nk = self.hint_loss(inputs,outputs)

        exist_preds = torch.clamp(outputs[OUTPUT_LANE_EXIST].sigmoid_(), min=1e-4, max=1 - 1e-4)
        segementation_pred_xs = outputs[OUTPUT_LANE_SEG_X]
        segementation_pred_ys = outputs[OUTPUT_LANE_SEG_Y]
        regression_preds = outputs[OUTPUT_LANE_REG_XY]

        gt_lanes = inputs[COLLECTION_GT_LANE_SEG]

        batch_size = exist_preds.shape[0]
        exist_loss_list = []
        segmentation_loss_list = []
        regression_loss_list = []

        exist_layer_radius = 4

        for j in range(batch_size):
            gt_lane = gt_lanes[j]
            gt_lane_ct, gt_lane_segmentation_x, gt_lane_segmentation_y, gt_lane_regression, gt_lane_mask = self.build_target(gt_lane)
            
            exist_pred = exist_preds[j]
            segementation_pred_x = segementation_pred_xs[j].unsqueeze(0)
            segementation_pred_y = segementation_pred_ys[j].unsqueeze(0)
            regression_pred = regression_preds[j].unsqueeze(0)

            exist_ints = gt_lane_ct.to(torch.int)
            exist_cls, output_h, output_w = exist_pred.shape
            exist_heatmap = torch.zeros((exist_cls, output_h, output_w), dtype=torch.float32).to("cuda")
            for k in range(gt_lane_ct.shape[0]):
                draw_umich_gaussian(exist_heatmap[0], exist_ints[k], max(0, int(gaussian_radius((exist_layer_radius, exist_layer_radius)))))

            exist_loss = ct_focal_loss(exist_pred, exist_heatmap.detach())

            # segementation_pred_x = segementation_pred_x.permute(0, 2, 3, 1).contiguous().view(-1, 49).contiguous()
            # gt_lane_segmentation_x = gt_lane_segmentation_x.view(-1).contiguous()
            segmentation_loss = self.focal_loss_x(segementation_pred_x, gt_lane_segmentation_x, class_weight=self.x_weights)

            # segementation_pred_y = segementation_pred_y.permute(0, 2, 3, 1).contiguous().view(-1, 25).contiguous()
            # gt_lane_segmentation_y = gt_lane_segmentation_y.view(-1).contiguous()
            segmentation_loss += self.focal_loss_y(segementation_pred_y, gt_lane_segmentation_y, class_weight=self.y_weights)

            # off_loss = self.regress_loss(regression_pred, gt_lane_regression.detach()) 
            # off_loss = F.mse_loss(regression_pred, gt_lane_regression.detach())
            off_loss = F.l1_loss(regression_pred * gt_lane_mask, gt_lane_regression.detach() * gt_lane_mask, size_average=False)
            off_loss = off_loss / (gt_lane_mask.sum() + 1e-4)

            exist_loss_list.append(exist_loss)
            segmentation_loss_list.append(segmentation_loss)
            regression_loss_list.append(off_loss)

        loss_dict["loss_lane_pos"] = self.weight * torch.stack(exist_loss_list).mean(dim=0, keepdim=True)
        loss_dict["loss_lane_seg"] = self.weight * torch.stack(segmentation_loss_list).mean(dim=0, keepdim=True)
        loss_dict["loss_lane_reg"] = self.weight * torch.stack(regression_loss_list).mean(dim=0, keepdim=True)
        loss_dict["loss_lane_hint_bk"] = hint_loss_bk
        loss_dict["loss_lane_hint_nk"] = hint_loss_nk
        
        return loss_dict
    
    def hint_loss(self, inputs, outputs):
        
        hint_loss_bk = 0
        hint_loss_nk = 0

        teacher_features_nk = outputs[OUTPUT_TEACHER_FEATURE_AUG]
        student_features_nk = outputs[OUTPUT_STUDENT_FEATURE_AUG]

        teacher_features_bk = outputs[OUTPUT_TEACHER_BACKBONE_AUG]
        student_features_bk = outputs[OUTPUT_STUDENT_BACKBONE_AUG]

        for teacher_feat, student_feat,adaptation in zip(teacher_features_bk,student_features_bk,self.adaptations_bk):
            teacher_feat = teacher_feat.detach()
            student_feat = adaptation(student_feat)
            hint_loss_bk += nn.MSELoss(size_average=True)(teacher_feat,student_feat)

        for teacher_feat, student_feat,adaptation in zip(teacher_features_nk,student_features_nk,self.adaptations_nk):
            teacher_feat = teacher_feat.detach()
            student_feat = adaptation(student_feat)
            hint_loss_nk += nn.MSELoss(size_average=True)(teacher_feat,student_feat)

        return hint_loss_bk, hint_loss_nk  

    def regress_loss(self, regression, targets):
        regression_diff = torch.abs(targets - regression)
        regression_loss = torch.where(
            torch.le(regression_diff, 1.0 / 9.0),
            0.5 * 9.0 * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / 9.0
        )
        return regression_loss.mean()

    def simple_test(self, features):
        pred = self.forward_single(features)
        return pred

    def build_target(self, gt_lane):
        gt_lane_org = gt_lane.permute(1,2,0).squeeze()
        lane_label_segmentation_x = torch.zeros((self.seg_size[0]*self.seg_size[1])).int().cuda()
        lane_label_segmentation_y = torch.zeros((self.seg_size[0]*self.seg_size[1])).int().cuda()
        lane_label_regression = torch.zeros((self.seg_size[0]*self.seg_size[1], 2)).float().cuda()
        lane_label_mask = torch.zeros((self.seg_size[0]*self.seg_size[1], 2)).float().cuda()

        lane_y_org = torch.arange(self.img_size[0]).view(-1,1).contiguous().repeat(1,self.img_size[1])
        lane_x_org = torch.arange(self.img_size[1]).view(1,-1).contiguous().repeat(self.img_size[0],1)

        heat_down_ratio = self.img_size[0] // self.heat_size[0]
        seg_down_ratio = self.img_size[0] // self.seg_size[0]
        
        lane_center_list = []
        for lane_label in self.lane_range:
            lane_point_x_org = lane_x_org[gt_lane_org==lane_label].view(-1,1)
            lane_point_y_org = lane_y_org[gt_lane_org==lane_label].view(-1,1)
            
            if lane_point_x_org.shape[0] > 0:
                lane_point_org = torch.cat((lane_point_x_org, lane_point_y_org), 1).contiguous().float()
                lane_center_org = torch.mean(lane_point_org, axis=0).int()//heat_down_ratio

                lane_point_squeeze = lane_point_x_org//seg_down_ratio + lane_point_y_org//seg_down_ratio*self.seg_size[1]
                lane_label_segmentation_x[lane_point_squeeze] = lane_center_org[0]+1
                lane_label_segmentation_y[lane_point_squeeze] = lane_center_org[1]+1
                lane_label_regression[lane_point_squeeze, 0] = lane_point_x_org.cuda().float()/seg_down_ratio - lane_point_x_org.float().cuda()//seg_down_ratio
                lane_label_regression[lane_point_squeeze, 1] = lane_point_y_org.cuda().float()/seg_down_ratio - lane_point_y_org.float().cuda()//seg_down_ratio
                lane_label_mask[lane_point_squeeze] = 1
                lane_center_list.append(lane_center_org)

        lane_center_list = torch.stack(lane_center_list, 0)
        lane_label_segmentation_x = lane_label_segmentation_x.view(1, self.seg_size[0], self.seg_size[1]).long()
        lane_label_segmentation_y = lane_label_segmentation_y.view(1, self.seg_size[0], self.seg_size[1]).long()
        lane_label_regression = lane_label_regression.view(1, self.seg_size[0], self.seg_size[1], 2).permute(0, 3, 1, 2).contiguous()
        lane_label_mask = lane_label_mask.view(1, self.seg_size[0], self.seg_size[1], 2).permute(0, 3, 1, 2).contiguous()

        return lane_center_list, lane_label_segmentation_x, lane_label_segmentation_y, lane_label_regression, lane_label_mask
