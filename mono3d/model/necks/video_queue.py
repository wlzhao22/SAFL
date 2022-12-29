from dcn_v2 import DCNv2
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor 
from typing import List
from torch.nn.modules.batchnorm import BatchNorm2d

from torch.nn.modules.conv import Conv2d 
from mmdet3d.models import NECKS


@NECKS.register_module()
class VideoFeatureFusion(nn.Module):
    def __init__(self, channels_in: int, frames: int, repeat: int, conv_config=(3, 1, 1, 1)) -> None:
        super().__init__()
        self.dropblock = None 
        self.frames = frames 
        self.repeat = repeat 

        chi = channels_in * 2
        k, s, p, d = conv_config
        self.conv_in = nn.Sequential(
            nn.Conv2d(channels_in, chi, 3, 1, 1, bias=False),
            nn.BatchNorm2d(chi),
            nn.ReLU(inplace=True),
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(chi, channels_in, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(channels_in), 
            nn.ReLU(inplace=True),
        )

        for t in range(frames - 1):
            conv_alpha = nn.Conv2d(chi, 1, 3, 1, 1)
            conv_alpha.weight.data.zero_()
            conv_alpha.bias.data.zero_()
            conv_offset_mask = nn.Conv2d(
                chi, 
                k * k * 3, 
                3, 1, 1
            )
            conv_offset_mask.weight.data.zero_()
            torch.nn.init.normal_(conv_offset_mask.bias.data, mean=0.0, std=33.0)
            dcn = DCNv2(
                chi, 
                chi, 
                k, 
                stride=s, 
                padding=p,
                dilation=d,
            )
            setattr(self, 'conv_alpha_t{}'.format(t), conv_alpha)
            setattr(self, 'conv_offset_mask_t{}'.format(t), conv_offset_mask)
            setattr(self, 'dcn_t{}'.format(t), dcn)
        
        for r in range(repeat):
            node = nn.Sequential(
                Conv2d(chi, chi, 3, 1, 1, bias=False), 
                BatchNorm2d(chi), 
                nn.ReLU(inplace=True),
            )
            setattr(self, 'node_r{}'.format(r), node)

    def forward(self, features: List[Tensor]):
        features = [self.conv_in(f) for f in features]
        f0 = features[0]
        f = f0
        for r in range(self.repeat):
            for t in range(self.frames - 1):
                ft = features[t + 1]
                conv_alpha = getattr(self, 'conv_alpha_t{}'.format(t))
                conv_offset_mask = getattr(self, 'conv_offset_mask_t{}'.format(t))
                dcn = getattr(self, 'dcn_t{}'.format(t))
                out = conv_offset_mask(f)
                o1, o2, mask = torch.chunk(out, 3, dim=1)
                offset = torch.cat((o1, o2), dim=1).mul(t + 1)
                mask = torch.sigmoid(mask) 
                if self.dropblock is not None:
                    ft = self.dropblock(ft)
                alpha = conv_alpha(f).sigmoid()
                f = dcn(ft, offset, mask) * alpha + f * (1 - alpha)
            
            node = getattr(self, 'node_r{}'.format(r))
            f = F.relu(node(f) + f0, inplace=True)
        return self.conv_out(f)



