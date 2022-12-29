from dcn_v2 import DCNv2, DCN
from numpy.core.shape_base import block
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor 
from typing import List
from torch.nn.modules.batchnorm import BatchNorm2d

from torch.nn.modules.conv import Conv2d 
from mmdet3d.models import NECKS
from mono3d.ops.anab_v2 import ANAB_v2
from dropblock import DropBlock2D

@NECKS.register_module()
class VideoFeatureFusion3(nn.Module):
    def __init__(self, channels_in: int, frames: int, repeat: int, conv_config=(3, 1, 1, 1)) -> None:
        super().__init__()
        self.frames = frames 
        self.repeat = repeat 
        chi = channels_in * 2 
        k, s, p, d = conv_config

        # self.dropblock = DropBlock2D(
        #     block_size=13,
        #     drop_prob=0.1
        # )
        self.dropblock = nn.Identity()
        self.conv_in = nn.Sequential(
            nn.Conv2d(channels_in, chi, 4, 2, 1, bias=False),
            nn.BatchNorm2d(chi),
            nn.ReLU(inplace=True),
        )
        # self.conv_in = nn.Sequential(
        #     nn.Conv2d(channels_in, chi, 3, 1, 1, bias=False), 
        #     nn.BatchNorm2d(chi),
        #     nn.ReLU(inplace=True),
        # )
        self.conv_out = nn.Sequential(
            nn.ConvTranspose2d(chi, channels_in, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_in, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(channels_in), 
            nn.ReLU(inplace=True),
        )
        # self.conv_out = nn.Sequential(
        #     nn.ConvTranspose2d(chi, channels_in, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(channels_in),
        #     nn.Conv2d(channels_in, channels_in, 3, 1, 1, bias=False), 
        #     nn.BatchNorm2d(channels_in), 
        #     nn.ReLU(inplace=True),
        # )
        
        self.conv_adp = nn.Sequential(
            DCN(chi, chi, 3, 1, 1),
            nn.ReLU(inplace=True),
            DCN(chi, chi, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv_alpha = nn.Conv2d(chi, 1, 3, 1, 1)
        self.conv_alpha.weight.data.zero_()
        self.conv_alpha.bias.data.zero_()
        self.conv_offset_mask = nn.Sequential(
            nn.Conv2d(chi, chi, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(
            chi, 
            k * k * 3, 
            3, 1, 1
            )
        )
        # self.conv_offset_mask.weight.data.zero_()
        # self.conv_offset_mask.bias.data.zero_()
        self.conv_offset_uncertainty = nn.Conv2d(chi, 1, 3, 1, 1)
        self.dcn = DCNv2(
            chi, 
            chi, 
            k, 
            stride=s, 
            padding=p,
            dilation=d,
        )
        self.conv_fusion = ANAB_v2(ch=chi, psp_size=[1, 4, 8, 16,])
    
    def forward(self, features: List[Tensor]):
        assert len(features) == self.frames
        bs, ch, h, w = features[0].shape
        feature_ori_dbg = features[0]
        feature_ori = self.dropblock(features[0])
        features = [self.conv_in(self.dropblock(f)) for f in features]
        for r in range(self.repeat):
            for t in range(self.frames - 1):
                f_prev = features[-t - 1]
                f_current = features[-t - 2]
                f_add = self.conv_fusion(f_current, f_prev)
                # f_add = f_current + self.conv_adp(f_prev)
                # out = self.conv_offset_mask(f_add)
                # o1, o2, mask = torch.chunk(out, 3, dim=1)
                # offset = torch.cat((o1, o2), dim=1)
                # mask = torch.sigmoid(mask) 
                # f_align = self.dcn(f_prev, offset.detach(), mask)
                f_align = f_add
                alpha = self.conv_alpha(f_add).sigmoid()
                f_out = f_align * alpha + f_current * (1 - alpha)
                features[-t - 2] = f_out

        return self.conv_out(features[0]), self.conv_out(features[1]) # + feature_ori
