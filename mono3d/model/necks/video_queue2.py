from dcn_v2 import DCNv2, DCN
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor 
from typing import List
from torch.nn.modules.batchnorm import BatchNorm2d

from torch.nn.modules.conv import Conv2d 
from mmdet3d.models import NECKS


def offset_activation_function(x):
    return torch.tan(torch.sigmoid(x) * 3.14 - 1.57)


@NECKS.register_module()
class VideoFeatureFusion2(nn.Module):
    def __init__(self, channels_in: int, frames: int, repeat: int, conv_config=(3, 1, 1, 1)) -> None:
        super().__init__()
        self.frames = frames 
        self.repeat = repeat 
        chi = channels_in * 2 
        k, s, p, d = conv_config

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
        self.conv_fusion = FusionModule(2, chi)
        # self.cheat = nn.Parameter(torch.zeros(2, 48, 160) + 0.5)
    
    def forward(self, features: List[Tensor]):
        assert len(features) == self.frames
        bs, ch, h, w = features[0].shape
        ret_offset_list: List[Tensor] = [None] * (self.frames - 1)
        ret_offset_uncertainty_list: List[Tensor] = [None] * (self.frames - 1)
        feature_ori = features[0]
        features = [self.conv_in(f) for f in features]
        for r in range(self.repeat):
            for t in range(self.frames - 1):
                f_prev = features[-t - 1]
                f_current = features[-t - 2]
                f_add = self.conv_fusion(f_current, f_prev)
                # f_add = f_current + self.conv_adp(f_prev)
                out = self.conv_offset_mask(f_add)
                o1, o2, mask = torch.chunk(out, 3, dim=1)
                offset = torch.cat((o1, o2), dim=1)
                mask = torch.sigmoid(mask) 
                f_align = self.dcn(f_prev, offset.detach(), mask)
                alpha = self.conv_alpha(f_add).sigmoid()
                f_out = f_align * alpha + f_current * (1 - alpha)
                features[-t - 2] = f_out
                if r == self.repeat - 1:
                    ret_offset_list[-t - 1] = offset
                    ret_offset_uncertainty_list[-t - 1] = self.conv_offset_uncertainty(f_add)

        ret_offset_list = [offset_activation_function(t.view(bs, 2, -1, h // 2, w // 2).mean(2)) if t is not None else None for t in ret_offset_list]
        # ret_offset_list = [torch.stack([offset_activation_function(self.cheat)] * bs) for t in ret_offset_list]
        return self.conv_out(features[0]) + feature_ori, ret_offset_list, ret_offset_uncertainty_list


class FusionModule(nn.Module):
    def __init__(self, n_level, channels_in) -> None:
        super().__init__()
        self.n_level = n_level
        for i in range(n_level):
            ch1 = channels_in * 2 ** i
            ch2 = channels_in * 2 ** (i + 1)
            conv_down = nn.Sequential(
                DCN(ch1, ch2, 4, 2, 1), 
                nn.BatchNorm2d(ch2),
                nn.ReLU(inplace=True),
            )
            # conv_up = nn.Sequential(
            #     nn.ConvTranspose2d(ch2, ch1, 4, 2, 1), 
            #     nn.BatchNorm2d(ch1), 
            #     nn.ReLU(inplace=True),
            # )
            conv_up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                DCN(ch2, ch1, 3, 1, 1),
                nn.ReLU(inplace=True)
            )
            setattr(self, 'conv_down_{}'.format(i), conv_down)
            setattr(self, 'conv_up_{}'.format(i), conv_up)
    
    def forward(self, a, b):
        f = a + b 
        return self.recur(f, 0)

    def recur(self, f, i):
        if i == self.n_level: return f
        conv_down = getattr(self, 'conv_down_{}'.format(i))
        conv_up = getattr(self, 'conv_up_{}'.format(i))
        f_down = conv_down(f)
        f_down = self.recur(f_down, i + 1)
        f_down_up = conv_up(f_down)
        return F.relu(f + f_down_up, inplace=True)
