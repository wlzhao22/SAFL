import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import kaiming_init
from mmdet3d.models.builder import HEADS
from abc import ABCMeta
from mmdet3d.models.utils.utils_2d.key_config import *
BatchNorm2d = nn.SyncBatchNorm
bn_mom = 0.1

@HEADS.register_module()
class LaneHeadPrune(nn.Module, metaclass=ABCMeta):
    def __init__(self,
                 row_anchor=None,
                 layers=[2,2,2,2],
                 segmentation_classes=4,
                 spp_planes=128,
                 head_planes=64,
                 planes=128):
        super(LaneHeadPrune, self).__init__()

        pruned_planes_ = [96, 168, 168, 120, 120, 120, 120, 96, 144, 144, 168, 384]
        pruned_planes = [96, 288, 312, 144, 288, 408, 624, 288, 720, 576, 648, 1536]

        highres_planes = planes * 2

        block = BasicBlock
        
        self.relu = nn.ReLU(inplace=False)

        weights = [2.0 for _ in range(segmentation_classes+1)]
        weights[0] = 0.4
        class_weights = torch.FloatTensor(weights).cuda()
        self.criterion_bi = torch.nn.BCEWithLogitsLoss().cuda()
        self.criterion_seg = torch.nn.BCEWithLogitsLoss().cuda()
        self.criterion_exist = torch.nn.BCEWithLogitsLoss().cuda()


        self.compression3 = nn.Sequential(
                                          nn.Conv2d(pruned_planes[4], pruned_planes_[4], kernel_size=1, bias=False),
                                          BatchNorm2d(pruned_planes_[4], momentum=bn_mom),
                                        )
        self.compression4 = nn.Sequential(
                                          nn.Conv2d(pruned_planes[8], pruned_planes_[8], kernel_size=1, bias=False),
                                          BatchNorm2d(pruned_planes_[8], momentum=bn_mom),
                                        )
        self.down3 = nn.Sequential(
                                  nn.Conv2d(pruned_planes_[4], pruned_planes[4], kernel_size=3, stride=2, padding=1, bias=False),
                                  BatchNorm2d(pruned_planes[4], momentum=bn_mom),
                                )
        self.down4 = nn.Sequential(
                                   nn.Conv2d(pruned_planes_[8], pruned_planes[8]//4, kernel_size=1, bias=False),
                                   BatchNorm2d(pruned_planes[8]//4, momentum=bn_mom),
                                   nn.Conv2d(pruned_planes[8]//4, pruned_planes[8]//2, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(pruned_planes[8]//2, momentum=bn_mom),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(pruned_planes[8]//2, pruned_planes[8], kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(pruned_planes[8], momentum=bn_mom),
                                   )

        # self.layer3_ = self._make_layer(block, planes, highres_planes, 2)
        self.layer3_ = self._make_layer(block, pruned_planes_[0], pruned_planes_[1:5], layers[0], shortcut=True)

        # self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)
        self.layer4_ = self._make_layer(block, pruned_planes_[4], pruned_planes_[5:9], layers[1])

        # self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)
        self.layer5_ = self._make_layer5(Bottleneck5, pruned_planes_[8], pruned_planes_[9:], 1)
        
        # self.layer3 = self._make_layer(block, planes, planes * 4, layers[2], stride=2)
        self.layer3 = self._make_layer(block, pruned_planes[0], pruned_planes[1:5], layers[2], stride=2, shortcut=True)
        
        # self.layer4 = self._make_layer4(block, planes * 4, planes * 8, layers[3], stride=2)
        self.layer4 = self._make_layer(block, pruned_planes[4], pruned_planes[5:9], layers[3], stride=2, shortcut=True)

        # self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)
        self.layer5 = self._make_layer5(Bottleneck5, pruned_planes[8], pruned_planes[9:], 1, stride=2)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        self.segmentation_head = self.build_segmentation_head(inplanes=planes*4, interplanes=head_planes,
                                                        outplanes=2,
                                                        kernel_size=1, upsampling=8)

        self.aux_head = self.build_segmentation_head(inplanes=pruned_planes_[3], interplanes=head_planes,
                                                        outplanes=2,
                                                        kernel_size=1, upsampling=8)

        self.segmentation_classes = segmentation_classes

        
    # def _make_layer(self, block, inplanes, planes, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride, bias=False),
    #             nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
    #         )

    #     layers = []
    #     layers.append(block(inplanes, planes, stride, downsample))
    #     inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         if i == (blocks-1):
    #             layers.append(block(inplanes, planes, stride=1, no_relu=True))
    #         else:
    #             layers.append(block(inplanes, planes, stride=1, no_relu=False))

    #     return nn.Sequential(*layers)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1, shortcut=False):

        planes1 = [planes[0], planes[1]]
        planes2 = [planes[2], planes[3]]

        downsample = None
        if shortcut:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes1[-1],
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes1[-1], momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes1, stride, downsample))
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(planes1[-1], planes2, stride=1, no_relu=True))
            else:
                layers.append(block(planes1[-1], planes2, stride=1, no_relu=False))

        return nn.Sequential(*layers)


    def _make_layer5(self, block, inplanes, planes, blocks, stride=1):

        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes[-1],
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes[-1], momentum=bn_mom),
        )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
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


    def build_segmentation_head(self, inplanes, interplanes, outplanes, kernel_size=3, upsampling=1):
        bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        upsampling = nn.Upsample(scale_factor=upsampling,mode='bilinear') if upsampling > 1 else nn.Identity()
        return nn.Sequential(conv1,bn2,relu,conv2,upsampling)
        return nn.Sequential(conv1,bn2,relu,conv2)

    def build_lane_exist(self, in_channels, out_channels, kernel_size):
        conv1 = nn.Conv2d(in_channels, in_channels//4, kernel_size=kernel_size, stride=2)
        bn1 = BatchNorm2d(in_channels//4, momentum=bn_mom)
        conv2 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=1, stride=1)
        maxpool = nn.MaxPool2d(2, stride=2)
        linear1 = nn.Linear(192, in_channels)  # 1/8
        linear2 = nn.Linear(in_channels, out_channels)
        relu = nn.ReLU()
        stage_1 = nn.Sequential(conv1, bn1, relu, conv2, maxpool)
        stage_2 = nn.Sequential(linear1, relu, linear2)
        return stage_1, stage_2

    def forward_single(self, features, deploy=True):
        output = {}
        layers = []
        x = features[1]  # 1/8
        _, _, h, w = x.shape

        x_ = self.layer3_(self.relu(x))
        x = self.layer3(self.relu(x))
        layers.append(x)

        x = x + self.down3(self.relu(x_))
        x_ = x_ + F.interpolate(
                        self.compression3(self.relu(layers[0])),
                        size=[h, w],
                        mode='bilinear')
        
        if not deploy:
            aux_out = self.aux_head(x_)
        
        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
                        self.compression4(self.relu(layers[1])),
                        size=[h, w],
                        mode='bilinear')
        
        x_ = self.layer5_(self.relu(x_))

        out_dappm, out_sum = self.spp(self.layer5(self.relu(x)))
        x = F.interpolate(
                    out_sum,
                    size=[h, w],
                    mode='bilinear')

        output[OUTPUT_LANE_SEG] = self.segmentation_head(x + x_)

        if not deploy:
            return output, aux_out
        else:
            output[OUTPUT_LANE_SEG] = output[OUTPUT_LANE_SEG].sigmoid()
            
            return output

    def forward(self, features):
        return self.forward_single(features)

    def forward_train(self, inputs, outputs):
        gt_mask = inputs[COLLECTION_GT_LANE_SEG].squeeze(1).long()
        
        features = outputs[OUTPUT_FEATURE_AUG]
        outputs = self.forward_single(features, deploy=False)
        
        pred_seg = outputs[0][OUTPUT_LANE_SEG]
        pred_aux = outputs[1]
        
        detect_loss = self.compute_loss(pred_seg, pred_aux, gt_mask)
        return detect_loss

    def compute_loss(self, pred, pred_aux, gt_seg):
        loss_dict = {}
        gt_seg_mask = torch.zeros_like(gt_seg)
        gt_seg_mask[gt_seg>0] = 1
        gt_seg_mask = gt_seg_mask.unsqueeze(1)

        gt_seg_mask = torch.nn.functional.one_hot(gt_seg_mask, num_classes=2).squeeze()
        gt_seg_mask = gt_seg_mask.permute(0,3,1,2).float().contiguous()

        loss_dict['loss_bi_seg'] = self.criterion_bi(pred, gt_seg_mask) + self.criterion_bi(pred_aux, gt_seg_mask)*0.4
        
        return loss_dict
    
    def simple_test(self, features):
        pred = self.forward_single(features)
        return pred


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes[0], stride)
        self.bn1 = BatchNorm2d(planes[0], momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes[0], planes[1])
        self.bn2 = BatchNorm2d(planes[1], momentum=bn_mom)
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
        
        # print('out: ', out.size())
        # print('residual: ', residual.size())

            out += residual

        # print('final out: ', out.size())

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck5(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck5, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes[0], momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes[1], momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes[1], planes[2], kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes[2], momentum=bn_mom)
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

# class Bottleneck(nn.Module):
#     expansion = 2

#     def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
#                                bias=False)
#         self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.no_relu = no_relu

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         if self.no_relu:
#             return out
#         else:
#             return self.relu(out)

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

        return out_dappm, out



