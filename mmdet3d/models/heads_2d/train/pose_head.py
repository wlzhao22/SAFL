import torch
from mmcv.cnn import xavier_init
from mmdet3d.models.builder import HEADS
import numpy as np
from abc import ABCMeta

import torch.nn as nn

@HEADS.register_module()
class PoseHead(nn.Module, metaclass=ABCMeta):

    def __init__(self,
                 # num_classes=2,
                 # in_channels=512,
                 # num_ch_enc=[6, 24, 40, 112, 1280],
                 num_ch_enc=[64, 128, 256, 512],
                 stride=1,
                 train_cfg=None,
                 test_cfg=None,
                 fix_para=False,):
        super(PoseHead, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_ch_enc = np.array(num_ch_enc)
        self.stride = stride
        self.init_layers()

    def init_layers(self):
        self.reduce = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.conv1 = nn.Conv2d(256, 256, 3, self.stride, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, self.stride, 1)
        self.conv3 = nn.Conv2d(256, 6, 1)

        self.relu = nn.ReLU()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, input_features, frame_id=0):
        f = input_features[-1]
        out = self.relu(self.reduce(f))
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)

        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, 1, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]
        
        return axisangle, translation


    def transformation_from_parameters(self, axisangle, translation, invert=False):
        R = self.rot_from_axisangle(axisangle)
        t = translation.clone()
        if invert:
            R = R.transpose(1, 2)
            t *= -1
        T = self.get_translation_matrix(t)
        if invert:
            M = torch.matmul(R, T)
        else:
            M = torch.matmul(T, R)
        return M

    def get_translation_matrix(self, translation_vector):
        T = torch.zeros(translation_vector.shape[0], 4, 4).cuda()
        t = translation_vector.contiguous().view(-1, 3, 1)
        T[:, 0, 0] = 1
        T[:, 1, 1] = 1
        T[:, 2, 2] = 1
        T[:, 3, 3] = 1
        T[:, :3, 3, None] = t
        return T

    def rot_from_axisangle(self, vec):
        angle = torch.norm(vec, 2, 2, True)
        axis = vec / (angle + 1e-7)
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        C = 1 - ca
        x = axis[..., 0].unsqueeze(1)
        y = axis[..., 1].unsqueeze(1)
        z = axis[..., 2].unsqueeze(1)
        xs = x * sa
        ys = y * sa
        zs = z * sa
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC
        rot = torch.zeros((vec.shape[0], 4, 4)).cuda()
        rot[:, 0, 0] = torch.squeeze(x * xC + ca)
        rot[:, 0, 1] = torch.squeeze(xyC - zs)
        rot[:, 0, 2] = torch.squeeze(zxC + ys)
        rot[:, 1, 0] = torch.squeeze(xyC + zs)
        rot[:, 1, 1] = torch.squeeze(y * yC + ca)
        rot[:, 1, 2] = torch.squeeze(yzC - xs)
        rot[:, 2, 0] = torch.squeeze(zxC - ys)
        rot[:, 2, 1] = torch.squeeze(yzC + xs)
        rot[:, 2, 2] = torch.squeeze(z * zC + ca)
        rot[:, 3, 3] = 1
        return rot

    def compute_losses(self, inputs, outputs):
        gt_pose_1, gt_pose_2 = inputs[COLLECTION_GT_POSE_1], inputs[COLLECTION_GT_POSE_2]
        pred_pose_1, pred_pose_2 = outputs[OUTPUT_POSE_PRE_TO_CUR], outputs[OUTPUT_POSE_CUR_TO_POST] 
        
        # Calculate velocity supervision loss
        loss = self.velocity_weight * sum([(gt_pose_1 - pred_pose_1).abs().mean(), (gt_pose_2 - pred_pose_2).abs().mean()])/2 
        return {"pose_loss": loss}

    def forward_train(self, inputs, outputs):

        return {}
        # return self.compute_losses(inputs, outputs)