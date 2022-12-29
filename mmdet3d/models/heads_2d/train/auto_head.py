import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmdet3d.models.builder import HEADS
import torch.nn.functional as F
import numpy as np
from mmdet3d.models.utils.utils_2d.layers import Conv1x1, ConvBlock, Conv3x3, SSIM
from abc import ABCMeta, abstractmethod
from mmdet3d.models.utils.utils_2d.key_config import *
import math
BatchNorm = nn.BatchNorm2d

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):
    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(
                    nn.Conv2d(c, out_dim,
                              kernel_size=1, stride=1, bias=False),
                    BatchNorm(out_dim),
                    nn.ReLU(inplace=True))
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(
                    out_dim, out_dim, f * 2, stride=f, padding=f // 2,
                    output_padding=0, groups=out_dim, bias=False)
                fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim,
                          kernel_size=node_kernel, stride=1,
                          padding=node_kernel // 2, bias=False),
                BatchNorm(out_dim),
                nn.ReLU(inplace=True))
            setattr(self, 'node_' + str(i), node)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
        return x, y


class DLAUp(nn.Module):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(3, channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])
            # print(f"{i}: {x.shape}, {len(y)}")
            layers[-i - 1:] = y
        return x

@HEADS.register_module()
class AutoHead(nn.Module, metaclass=ABCMeta):

    def __init__(self,
                 mean,
                 std,
                 auto_weight,
                 num_ch_enc=[64, 64, 128, 256, 512],
                 num_output_channels=3,
                 train_cfg=None,
                 test_cfg=None,
                 fix_para=False,):
        super(AutoHead, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_ch_enc = np.array(num_ch_enc)

        self.mean = torch.from_numpy(np.float64(mean).reshape(1, -1)).view(1, -1, 1, 1).cuda()
        self.stdinv = torch.from_numpy(1 / np.float64(std).reshape(1, -1)).view(1, -1, 1, 1).cuda()
        self.num_output_channels = num_output_channels
        self.auto_weight = auto_weight

        self.ssim = SSIM()

        self.init_layers()
        self.init_weights()

    def init_layers(self):
        self.first_level = int(np.log2(1))
        channels = [16, 32, 64, 128, 256, 512]
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)

        # disp
        self.disp0 = nn.Sequential(Conv3x3(channels[0], 3), nn.Sigmoid())


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward_single(self, feats):
        outputs = {}
        x = self.dla_up(feats[self.first_level:])
        dp0 = self.disp0(x)

        outputs[OUTPUT_AUTO_SCALE_0] = torch.mul(dp0*255 - self.mean, self.stdinv)
        outputs[OUTPUT_AUTO_IMG] = dp0*255

        return outputs

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.
        """
        # return multi_apply(self.forward_single, feats)
        return self.forward_single(feats)

    def compute_losses(self, inputs, outputs):
        loss_dict = {}
        target = inputs[COLLECTION_IMG]
        for index, scale in enumerate(OUTPUT_AUTO_SCALES[:1]):
            res_img = outputs[scale]
            _, _, h, w = res_img.size()
            target_resize = F.interpolate(target, [h, w], mode="bilinear", align_corners=False)
            img_reconstruct_loss = self.compute_reprojection_loss(res_img, target_resize)
            loss_dict['img_reconstruct_loss_' + str(index)] = self.auto_weight * img_reconstruct_loss.mean() / len(OUTPUT_AUTO_SCALES)

        # for index, f in enumerate(outputs[OUTPUT_AUTO_FEATURE]):
        #     regularization_loss = self.get_feature_regularization_loss(f, target)
        #     loss_dict['feature_regularization_loss' + str(index)] = self.auto_weight * regularization_loss/(2 ** index)/1

        return loss_dict

    def robust_l1(self, pred, target):
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

    def compute_reprojection_loss(self, pred, target):
        photometric_loss = self.robust_l1(pred, target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss


    def forward_train(self,
                      inputs,
                      outputs):

        outputs.update(self.forward_single(outputs[OUTPUT_AUTO_FEATURE]))
        losses = self.compute_losses(inputs, outputs)

        return losses

    def gradient(self, D):
        D_dy = D[:, :, 1:] - D[:, :, :-1]
        D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return D_dx, D_dy

    def get_feature_regularization_loss(self, feature, img):
        b, _, h, w = feature.size()
        img = F.interpolate(img, (h, w), mode='area')

        feature_dx, feature_dy = self.gradient(feature)
        img_dx, img_dy = self.gradient(img)

        feature_dxx, feature_dxy = self.gradient(feature_dx)
        feature_dyx, feature_dyy = self.gradient(feature_dy)

        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)

        smooth1 = torch.mean(feature_dx.abs() * torch.exp(-img_dx.abs().mean(1, True))) + \
                  torch.mean(feature_dy.abs() * torch.exp(-img_dy.abs().mean(1, True)))

        smooth2 = torch.mean(feature_dxx.abs() * torch.exp(-img_dxx.abs().mean(1, True))) + \
                  torch.mean(feature_dxy.abs() * torch.exp(-img_dxy.abs().mean(1, True))) + \
                  torch.mean(feature_dyx.abs() * torch.exp(-img_dyx.abs().mean(1, True))) + \
                  torch.mean(feature_dyy.abs() * torch.exp(-img_dyy.abs().mean(1, True)))

        return -0.001 * smooth1+ 0.001 * smooth2



