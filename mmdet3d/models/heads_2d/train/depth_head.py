import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, kaiming_init
from mmdet.models.builder import HEADS
import torch.nn.functional as F
import numpy as np
from mmdet3d.models.utils.utils_2d.layers import Conv1x1, Conv3x3, CRPBlock, upsample, Backproject, Project, SSIM, ConvBlock
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
        res = []
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])
            # print(f"{i}: {x.shape}, {len(y)}")
            layers[-i - 1:] = y
            res.append(x)
        return res


@HEADS.register_module()
class DepthHead(nn.Module, metaclass=ABCMeta):

    def __init__(self,
                 imgs_per_gpu,
                 height,
                 width,
                 min_depth,
                 max_depth,
                 automask,
                 disp_norm,
                 smoothness_weight,
                 perception_weight,
                 reconstruct_weight,
                 num_ch_enc=[64, 64, 128, 256, 512],
                 scale_num = 4,
                 fix_para=False,):
        super(DepthHead, self).__init__()

        self.height, self.width = height, width
        self.min_depth, self.max_depth = min_depth, max_depth

        self.imgs_per_gpu = imgs_per_gpu
        self.automask = automask

        self.num_ch_enc = np.array(num_ch_enc)
        self.scale_num = scale_num

        self.disp_norm = disp_norm
        self.smoothness_weight = smoothness_weight
        self.perception_weight = perception_weight
        self.reconstruct_weight = reconstruct_weight

        self.backproject = Backproject(self.imgs_per_gpu, self.height, self.width)
        self.project = Project(self.imgs_per_gpu, self.height, self.width)
        self.ssim = SSIM()


        self.init_layers()
        self.init_weights()

    def init_layers(self):
        self.first_level = int(np.log2(1))
        channels = [16, 32, 64, 128, 256, 512]
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)

        # disp
        self.disp3 = nn.Sequential(Conv3x3(channels[3], 1), nn.Sigmoid())
        self.disp2 = nn.Sequential(Conv3x3(channels[2], 1), nn.Sigmoid())
        self.disp1 = nn.Sequential(Conv3x3(channels[1], 1), nn.Sigmoid())
        self.disp0 = nn.Sequential(Conv3x3(channels[0], 1), nn.Sigmoid())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward_single(self, feats):
        outputs = {}
        res = self.dla_up(feats[self.first_level:])
        dp3 = self.disp3(res[-4])
        dp2 = self.disp2(res[-3])
        dp1 = self.disp1(res[-2])
        dp0 = self.disp0(res[-1])


        # outputs[OUTPUT_DISP_SCALE_4] = dp4
        outputs[OUTPUT_DISP_SCALE_3] = dp3
        outputs[OUTPUT_DISP_SCALE_2] = dp2
        outputs[OUTPUT_DISP_SCALE_1] = dp1
        outputs[OUTPUT_DISP_SCALE_0] = dp0

        return outputs

    def forward(self, feats):
        return self.forward_single(feats)

    def forward_train(self,
                      inputs,
                      outputs,
                      **kwargs):

        outputs.update(self.forward_single(outputs[OUTPUT_FEATURE]))
        losses = self.compute_losses(inputs, outputs)

        return losses

    def compute_losses(self, inputs, outputs):
        loss_dict = {}

        # car_mask = inputs[COLLECTION_CAR_MASK]
        car_mask = 1

        target = inputs[COLLECTION_IMG]* car_mask

        scale_len = len(OUTPUT_DISP_SCALES[:self.scale_num])
        outputs = self.generate_features_pred(inputs, outputs)
        for scale in range(scale_len):
            """
            initialization
            """
            disp = outputs[OUTPUT_DISP_SCALES[scale]]
            # car_mask_disp = F.interpolate(car_mask, disp.shape[2:], mode="bilinear", align_corners=False)
            # car_mask_scale = F.interpolate(car_mask, [int(self.height/4), int(self.width/4)], mode="bilinear", align_corners=False)
            car_mask_disp = 1
            car_mask_scale = 1
            disp = car_mask_disp * disp

            reprojection_losses = []
            perceptional_losses = []
            
            """
            reconstruction
            """
            outputs = self.generate_images_pred(inputs, outputs, scale)

            """
            automask
            """
            if self.automask:
                for k in COLLECTION_IMGS[1:]:
                    # pred = inputs[k] * car_mask
                    pred = inputs[k]
                    identity_reprojection_loss = self.compute_reprojection_loss(pred, target)
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 1e-5
                    reprojection_losses.append(identity_reprojection_loss)
            
            """
            minimum reconstruction loss
            """
            for frame_id in range(len(COLLECTION_IMGS[1:])):
                # pred = outputs[("color", frame_id, scale)] * car_mask
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_loss = torch.cat(reprojection_losses, 1)

            min_reconstruct_loss, _ = torch.min(reprojection_loss, dim=1)
            loss_dict['loss_min_reconstruct_' + str(scale)] = self.reconstruct_weight * min_reconstruct_loss.mean()/scale_len

            # """
            # minimum perceptional loss
            # """
            for resample_key in OUTPUT_AUTO_FEATURE_RESAMPLES:
                # src_f = outputs[resample_key] * car_mask_scale
                # tgt_f = outputs[OUTPUT_AUTO_FEATURE][0] * car_mask_scale
                src_f = outputs[resample_key]
                tgt_f = outputs[OUTPUT_AUTO_FEATURE][0]
                perceptional_losses.append(self.compute_perceptional_loss(tgt_f, src_f))
            perceptional_loss = torch.cat(perceptional_losses, 1)

            min_perceptional_loss, _ = torch.min(perceptional_loss, dim=1)
            loss_dict['loss_min_perceptional_' + str(scale)] = self.perception_weight * min_perceptional_loss.mean() / scale_len

            """
            disp mean normalization
            """
            if self.disp_norm:
                mean_disp = disp.mean(2, True).mean(3, True)
                disp = disp / (mean_disp + 1e-7)

            """
            smooth loss
            """
            smooth_loss = self.get_smooth_loss(disp, target)
            loss_dict['loss_smooth_' + str(scale)] = self.smoothness_weight * smooth_loss / (2 ** scale)/scale_len

        return loss_dict

    def disp_to_depth(self, disp, min_depth=0.01, max_depth=100):
        min_disp = 1 / max_depth  
        max_disp = 1 / min_depth  
        scaled_disp = min_disp + (max_disp - min_disp) * disp  # (10-0.01)*disp+0.01
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def generate_images_pred(self, inputs, outputs, scale):
        K , inv_K = inputs[COLLECTION_SCALED_CALIB].float(), inputs[COLLECTION_SCALED_INV_CALIB].float()
        disp = outputs[OUTPUT_DISP_SCALES[scale]]
        disp = F.interpolate(disp, [self.height, self.width], mode="bilinear", align_corners=False)
        _, depth = self.disp_to_depth(disp, self.min_depth, self.max_depth)

        if scale == 0:
            print(depth)

        for frame_id, content in enumerate(zip(COLLECTION_IMGS[1:], OUTPUT_POSES)):
            img, pose = inputs[content[0]], outputs[content[1]]
            cam_points = self.backproject(depth, inv_K)
            pix_coords = self.project(cam_points, K, pose)
            outputs[("color", frame_id, scale)] = F.grid_sample(img, pix_coords, padding_mode="border")
        return outputs

    def generate_features_pred(self, inputs, outputs):
        K , inv_K = inputs[COLLECTION_SCALED_CALIB].float(), inputs[COLLECTION_SCALED_INV_CALIB].float()
        disp = outputs[OUTPUT_DISP_SCALES[0]]
        disp = F.interpolate(disp, [int(self.height), int(self.width)], mode="bilinear", align_corners=False)
        _, depth = self.disp_to_depth(disp, self.min_depth, self.max_depth)
        for frame_id, content in enumerate(zip(COLLECTION_IMGS[1:], OUTPUT_POSES, OUTPUT_AUTO_FEATURES[1:])):
            img, pose, features = inputs[content[0]], outputs[content[1]], outputs[content[2]]
            
            backproject = Backproject(self.imgs_per_gpu, int(self.height), int(self.width))
            project = Project(self.imgs_per_gpu, int(self.height), int(self.width))

            cam_points = backproject(depth, inv_K)
            pix_coords = project(cam_points, K, pose)
            outputs[OUTPUT_AUTO_FEATURE_RESAMPLES[frame_id]] = F.grid_sample(features[0], pix_coords, padding_mode="border")
        return outputs

    def robust_l1(self, pred, target):
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

    def compute_perceptional_loss(self, tgt_f, src_f):
        loss = self.robust_l1(tgt_f, src_f).mean(1, True)
        return loss

    def compute_reprojection_loss(self, pred, target):
        photometric_loss = self.robust_l1(pred, target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss

    def get_smooth_loss(self, disp, img):
        b, _, h, w = disp.size()
        a1 = 0.5
        a2 = 0.5
        img = F.interpolate(img, (h, w), mode='area')

        disp_dx, disp_dy = self.gradient(disp)
    
        img_dx, img_dy = self.gradient(img)

        disp_dxx, disp_dxy = self.gradient(disp_dx)
        disp_dyx, disp_dyy = self.gradient(disp_dy)

        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)

        smooth1 = torch.mean(disp_dx.abs() * torch.exp(-a1 * img_dx.abs().mean(1, True))) + \
                  torch.mean(disp_dy.abs() * torch.exp(-a1 * img_dy.abs().mean(1, True)))

        smooth2 = torch.mean(disp_dxx.abs() * torch.exp(-a2 * img_dxx.abs().mean(1, True))) + \
                  torch.mean(disp_dxy.abs() * torch.exp(-a2 * img_dxy.abs().mean(1, True))) + \
                  torch.mean(disp_dyx.abs() * torch.exp(-a2 * img_dyx.abs().mean(1, True))) + \
                  torch.mean(disp_dyy.abs() * torch.exp(-a2 * img_dyy.abs().mean(1, True)))

        return smooth1+smooth2

    
    def gradient(self, D):
        D_dy = D[:, :, 1:, :] - D[:, :, :-1, :]
        D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return D_dx, D_dy
