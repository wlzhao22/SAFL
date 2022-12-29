import torch 
import torch.nn as nn 
from dcn_v2 import DCNv2
from typing import List
from mmdet3d.models import NECKS

from torch.nn.modules import conv

from mmdet3d.ops.DCNv2.dcn_v2 import DCN 


@NECKS.register_module()
class DCNFPN(nn.Module):
    conv_configs = [
        (4, 2, 1, 1),
        (4, 4, 3, 3),
    ]

    def __init__(self, channels: List[int], repeat: int,
    ) -> None:
        super().__init__()
        levels = len(channels)
        self.levels = levels 
        self.repeat = repeat

        out_channels = channels[-1]
        for l in range(levels - 1):
            in_channels = channels[levels - 2 - l]
            k, s, p, d = self.conv_configs[l]
            assert s == 2**(l + 1)
            conv_offset_mask = nn.Conv2d(
                out_channels,
                k * k * 3, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                dilation=1,
                bias=True, 
            )
            # init offset 
            conv_offset_mask.weight.data.zero_() 
            conv_offset_mask.bias.data.zero_()
            dcn = DCNv2(
                    in_channels, 
                    out_channels, 
                    k, 
                    stride=s,
                    padding=p,
                    dilation=d,
                )
            setattr(self, 'conv_offset_mask_l{}'.format(l), conv_offset_mask)
            setattr(self, 'dcn_l{}'.format(l), dcn)
        self.conv_res = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.conv_res.weight.data.zero_()
        self.conv_res.bias.data.zero_()

    def forward(self, features):
        fh = features[-1]  # high level feature map 
        f = fh 
        for i in range(self.repeat):
            for l in range(self.levels - 1):
                conv_offset_mask = getattr(self, 'conv_offset_mask_l{}'.format(l))
                dcn = getattr(self, 'dcn_l{}'.format(l))
                
                out = conv_offset_mask(f)
                o1, o2, mask = torch.chunk(out, 3, dim=1)
                offset = torch.cat((o1, o2), dim=1)
                mask = torch.sigmoid(mask) 
                f = dcn(features[self.levels - 2 - l], offset, mask).relu() + f
        res = self.conv_res(f)
        return fh.add(res) 
