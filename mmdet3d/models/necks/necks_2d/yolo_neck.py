import torch.nn as nn
import torch
import math
from ...builder import NECKS


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, activation=None, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    if activation:
        result.add_module('avtivation', activation)
    return result

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = conv_bn(c1, c_, 1, 1, 0, nn.ReLU())
        self.cv2 = conv_bn(c_, c2, 3, 1, 1, nn.ReLU())
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = conv_bn(c1, c_, 1, 1, 0, nn.ReLU())
        self.cv2 = conv_bn(c1, c_, 1, 1, 0, nn.ReLU())
        self.cv3 = conv_bn(2 * c_, c2, 1, 1, 0, nn.ReLU())
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class YoloNeckBlock(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, cat_channels=512, n=1, c3_out_channels=None, is_upsample=True):
        super(YoloNeckBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        if c3_out_channels is None:
            c3_out_channels = out_channels
        self.is_upsample = is_upsample

        if self.is_upsample:
            self.conv = conv_bn(in_channels, out_channels, 1, 1, 0, nn.ReLU())
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.conv = conv_bn(in_channels, out_channels, 3, 2, 1, nn.ReLU())
        self.c3 = C3(out_channels+cat_channels, out_channels, n)

    def forward(self, x1, x2):
        out_1 = self.conv(x1)
        out_2 = torch.cat((self.upsample(out_1) if self.is_upsample else out_1, x2), 1)
        out_3 = self.c3(out_2)
        return out_1, out_2, out_3

def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor

@NECKS.register_module()
class YoloNeck(nn.Module):
    def __init__(self, depth_multiple, width_multiple, in_channels, out_channels=[512, 256, 256, 512], bottle_depths=[3, 3, 3, 3]):
        super(YoloNeck, self).__init__()
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.bottle_depths = bottle_depths
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.concats = []
        self.stage_channels = [in_channels[1], in_channels[2], make_divisible(self.out_channels[1] * self.width_multiple, 8), make_divisible(self.out_channels[0] * self.width_multiple, 8)]
        self.init_model()

    def init_model(self):
        self.fpn = nn.ModuleList()
        c1 = self.in_channels[0]
        for idx, depth in enumerate(self.bottle_depths):
            c2 = make_divisible(self.out_channels[idx] * self.width_multiple, 8)
            self.fpn.append(YoloNeckBlock(c1, c2, self.stage_channels[idx], depth, None if idx < 3 else c2 * 2, True if idx < 2 else False))
            c1 = c2

    def forward(self, features):
        s1_out_1, s1_out_2, s1_out_3 = self.fpn[0](features[2], features[1])
        s2_out_1, s2_out_2, s2_out_3 = self.fpn[1](s1_out_3, features[0])
        s3_out_1, s3_out_2, s3_out_3 = self.fpn[2](s2_out_3, s2_out_1)
        s4_out_1, s4_out_2, s4_out_3 = self.fpn[3](s3_out_3, s1_out_1)
        return s2_out_3, s3_out_3, s4_out_3