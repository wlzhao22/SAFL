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
class FreeSpaceHeadStu(nn.Module, metaclass=ABCMeta):
    def __init__(self,
                 layers=[1,1,1,1],
                 planes=24,
                #  planes=32,
                 inplanes=96,
                 spp_planes=48,
                #  spp_planes=64,
                 head_planes=24,):
                # head_planes=32):
        super(FreeSpaceHeadStu, self).__init__()

        highres_planes = planes * 2
        block = BasicBlock
        self.relu = nn.ReLU(inplace=False)

        # self.criteria_freespace = LabelSmoothing(smoothing=0.1)
        
        self.layer3 = self._make_layer(block, inplanes, planes * 4, layers[2], stride=2)
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

        self.layer3_ = self._make_layer(block, inplanes, highres_planes, 2)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)

        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        self.freespace_head = self.build_head(inplanes=planes*4, interplanes=head_planes, outplanes=1)

        self.adaptations_bk= nn.ModuleList([
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 1280, kernel_size=1, stride=1, padding=0)])
        
     
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

        output[OUTPUT_FREE_SPACE] = self.freespace_head(x + x_)
        
        if not deploy:
            return output
        else:
            return output

    def forward(self, features):
        return self.forward_single(features)

    def forward_train(self, inputs, outputs):
        outputs.update(self.forward_single(outputs[OUTPUT_STUDENT_BACKBONE], deploy=False))
        loss_dict = {}

        loss_dict.update(self.compute_freespace_loss(outputs, inputs))
        
        return loss_dict

    def compute_freespace_loss(self, outputs, inputs):
        loss_dict = {}        
        gt_freespace = inputs[COLLECTION_GT_FREE_SPACE_RESIZED].squeeze(1).long()
        pred_freespace = outputs[OUTPUT_FREE_SPACE].squeeze(1)
        gt_cls = torch.argmax(gt_freespace,dim=1)
        teacher_pred = outputs[OUTPUT_TEACHER_PRED_FREESPACE].detach()
        # criteria = nn.CrossEntropyLoss()
        # loss_dict["loss_freespace_seg"] = criteria(pred_freespace,gt_cls)
        # loss_dict["loss_freespace_seg"] = self.criteria_freespace(pred_freespace.permute(0,2,1), gt_cls)
        loss_dict["loss_freespace_distill"] = self.distillation(pred_freespace.squeeze(1), gt_cls, teacher_pred, temp=5.0, alpha=0.7)

        hint_loss_bk = self.hint_loss(inputs,outputs)
        loss_dict["loss_freespace_hint_bk"] = hint_loss_bk

        return loss_dict

    def distillation(self, y, labels, teacher_scores, temp, alpha):
        # return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (temp * temp * 2.0 * alpha) \
        #    + F.cross_entropy(y, labels) * (1. - alpha)

        return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (temp * temp * 2.0 * alpha) \
           + LabelSmoothing(smoothing=0.1)(y.permute(0,2,1), labels) * (1. - alpha)
    
    def hint_loss(self, inputs, outputs):
        
        hint_loss_bk = 0

        teacher_features_bk = outputs[OUTPUT_TEACHER_BACKBONE]
        student_features_bk = outputs[OUTPUT_STUDENT_BACKBONE]

        for teacher_feat, student_feat,adaptation in zip(teacher_features_bk,student_features_bk,self.adaptations_bk):
            teacher_feat = teacher_feat.detach()
            student_feat = adaptation(student_feat)
            hint_loss_bk += nn.MSELoss(size_average=True)(teacher_feat,student_feat)

        return hint_loss_bk
   
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




