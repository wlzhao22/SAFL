import torch
import numpy as np
import torch
import pdb
import numpy as np
from torch import nn
from torch.functional import norm
from torch.nn import functional as F
import torch.distributed as dist
import math
import time 

from mono3d.model.ops.drconv import DRConv2d_v2 

from .. import registry
from ..layers.utils import sigmoid_hm
from ..make_layers import _fill_fc_weights
from .get_targets import num_center_types
from mmcv.cnn import build_norm_layer

from inplace_abn import InPlaceABNSync, InPlaceABN, ABN


def build_abn(norm_cfg, num_features):
    norm_cfg = norm_cfg.copy()
    type = norm_cfg['type']
    if type.lower() == 'inplace_abn':
        norm_cfg.pop('type')
        if dist.is_available() and dist.is_initialized():
            return InPlaceABNSync(num_features, **norm_cfg)
        return InPlaceABN(num_features, **norm_cfg)
    else:
        return nn.Sequential(
            build_norm_layer(norm_cfg, num_features)[1],
            nn.LeakyReLU(inplace=True)
        )


@registry.PREDICTOR.register("Base_Predictor")
class _predictor(nn.Module):
    def __init__(self, 
        num_classes, 
        in_channels,
        input_width, 
        input_height, 
        reg_heads, 
        reg_channels, 
        down_ratio, 
        num_channel,
        norm_cfg,
        bn_momentum,
        init_p,
        uncertainty_init,
        enable_edge_fusion,
        edge_fusion_kernel_size,
        edge_fusion_norm,
        edge_fusion_relu,
        **kwargs
    ):
        super(_predictor, self).__init__()
        # ("Car", "Cyclist", "Pedestrian")
        classes = num_classes 

        self.bn_momentum = bn_momentum
        self.regression_head_cfg = reg_heads
        self.regression_channel_cfg = reg_channels
        self.output_width = input_width //  down_ratio
        self.output_height = input_height // down_ratio
        
        self.head_conv = num_channel

        ###########################################
        ###############  Cls Heads ################
        ########################################### 

        self.class_head = nn.Sequential(
            nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
            build_abn(norm_cfg, self.head_conv), 
            nn.Conv2d(self.head_conv, classes, kernel_size=1, padding=1 // 2, bias=True)
        )
        
        self.class_head[-1].bias.data.fill_(- np.log(1 / init_p - 1))

        ###########################################
        ############  Regression Heads ############
        ########################################### 
        
        # init regression heads
        self.reg_features = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        # init regression heads
        for idx, regress_head_key in enumerate(self.regression_head_cfg):
            feat_layer = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                build_abn(norm_cfg, self.head_conv)
            )
            
            self.reg_features.append(feat_layer)
            # init output head
            head_channels = self.regression_channel_cfg[idx]
            head_list = nn.ModuleList()
            for key_index, key in enumerate(regress_head_key):
                key_channel = head_channels[key_index]
                output_head = nn.Conv2d(self.head_conv, key_channel, kernel_size=1, padding=1 // 2, bias=True)

                if key.find('uncertainty') >= 0 and uncertainty_init:
                    torch.nn.init.xavier_normal_(output_head.weight, gain=0.01)
                
                # since the edge fusion is applied to the offset branch, we should save the index of this branch
                if key == '3d_offset': self.offset_index = [idx, key_index]

                _fill_fc_weights(output_head, 0)
                head_list.append(output_head)

            self.reg_heads.append(head_list)

        ###########################################
        ##############  Edge Feature ##############
        ###########################################

        # edge feature fusion
        self.enable_edge_fusion = enable_edge_fusion
        self.edge_fusion_kernel_size = edge_fusion_kernel_size
        self.edge_fusion_relu = edge_fusion_relu

        if self.enable_edge_fusion:
            trunc_norm_func = nn.BatchNorm1d if edge_fusion_norm == 'BN' else nn.Identity
            trunc_activision_func = nn.ReLU(inplace=True) if self.edge_fusion_relu else nn.Identity()
            
            self.trunc_heatmap_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size, padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                trunc_norm_func(self.head_conv, momentum=self.bn_momentum), trunc_activision_func, nn.Conv1d(self.head_conv, classes, kernel_size=1),
            )
            
            self.trunc_offset_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size, padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                trunc_norm_func(self.head_conv, momentum=self.bn_momentum), trunc_activision_func, nn.Conv1d(self.head_conv, 2, kernel_size=1),
            )

    def forward(self, inputs, outputs):
        # t = time.time()
        features = outputs['feature']
        b, c, h, w = features.shape

        if self.enable_edge_fusion:            
            edge_indices = inputs['edge_indices'] # B x K x 2
            edge_lens = inputs['edge_len'] # B
            
            # normalize
            grid_edge_indices = edge_indices.view(b, -1, 1, 2).float()
            grid_edge_indices[..., 0] = grid_edge_indices[..., 0] / (self.output_width - 1) * 2 - 1
            grid_edge_indices[..., 1] = grid_edge_indices[..., 1] / (self.output_height - 1) * 2 - 1

        feature_cls = self.class_head[:-1](features)
        output_cls = self.class_head[-1](feature_cls)

        # print('head pred cls: ', time.time() - t)
        # t = time.time() 
        output_regs = []
        # output regression
        for i, reg_feature_head in enumerate(self.reg_features):
            f = features
            reg_feature = reg_feature_head(f)

            for j, reg_output_head in enumerate(self.reg_heads[i]):
                output_reg = reg_output_head(reg_feature)

                # apply edge feature enhancement
                if self.enable_edge_fusion and i == self.offset_index[0] and j == self.offset_index[1]:
                    # apply edge fusion for both offset and heatmap
                    feature_for_fusion = torch.cat((feature_cls, reg_feature), dim=1)
                    edge_features = F.grid_sample(feature_for_fusion, grid_edge_indices, align_corners=True).squeeze(-1)

                    edge_cls_feature = edge_features[:, :self.head_conv, ...]
                    edge_offset_feature = edge_features[:, self.head_conv:, ...]
                    edge_cls_output = self.trunc_heatmap_conv(edge_cls_feature)
                    edge_offset_output = self.trunc_offset_conv(edge_offset_feature)
                    
                    for k in range(b):
                        edge_indice_k = edge_indices[k, :edge_lens[k].item()]
                        output_cls[k, :, edge_indice_k[:, 1], edge_indice_k[:, 0]] += edge_cls_output[k, :, :edge_lens[k]]
                        output_reg[k, :, edge_indice_k[:, 1], edge_indice_k[:, 0]] += edge_offset_output[k, :, :edge_lens[k]]
                
                output_regs.append(output_reg)

        output_cls = sigmoid_hm(output_cls)
        output_regs = torch.cat(output_regs, dim=1)

        ret = {'cls': output_cls, 'reg': output_regs}
        # print("head pred regression: ", time.time() - t)
        return ret 
        

class DynamicLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_cls: int, bias: bool=True):
        super().__init__()
        self.num_cls = num_cls
        self.weight = nn.Parameter(torch.Tensor(num_cls, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_cls, out_channels))
        else: 
            self.register_parameter('bias', None) 
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, category):
        weight = self.weight[category]  # shape = n, c_in, c_out 
        bias = self.bias[category] if self.bias is not None else 0  # shape = n, c_out 
        return torch.bmm(input.unsqueeze(1), weight).squeeze(1) + bias 


@registry.PREDICTOR.register("LazyPredictor")
class _lazy_predictor(_predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        in_channels = kwargs['in_channels']
        norm_cfg = kwargs['norm_cfg']
        uncertainty_init = kwargs['uncertainty_init']

            # init regression heads
        self.reg_features = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        # init regression heads
        for idx, regress_head_key in enumerate(self.regression_head_cfg):
            feat_layer = DynamicLinear(in_channels, self.head_conv, num_center_types)
            
            self.reg_features.append(feat_layer)
            # init output head
            head_channels = self.regression_channel_cfg[idx]
            head_list = nn.ModuleList()
            for key_index, key in enumerate(regress_head_key):
                key_channel = head_channels[key_index]
                output_head = DynamicLinear(self.head_conv, key_channel, num_center_types)

                if key.find('uncertainty') >= 0 and uncertainty_init:
                    torch.nn.init.xavier_normal_(output_head.weight, gain=0.01)
                
                # since the edge fusion is applied to the offset branch, we should save the index of this branch
                if key == '3d_offset': self.offset_index = [idx, key_index]

                _fill_fc_weights(output_head, 0)
                head_list.append(output_head)

            self.reg_heads.append(head_list)

    def forward(self, inputs, outputs):
        features = outputs['feature']
        b, c, h, w = features.shape

        # output classification
        feature_cls = self.class_head[:-1](features)
        output_cls = self.class_head[-1](feature_cls)
        output_cls = sigmoid_hm(output_cls)

        return {'cls': output_cls, 'reg': features}

    def forward2(self, features, center_types):
        output_regs = []
        # output regression
        for i, reg_feature_head in enumerate(self.reg_features):
            reg_feature = reg_feature_head(features, center_types).relu()

            for j, reg_output_head in enumerate(self.reg_heads[i]):
                output_reg = reg_output_head(reg_feature, center_types)
                output_regs.append(output_reg)

        output_regs = torch.cat(output_regs, dim=1)
        return output_regs

def make_predictor(lazy_regression=False, **cfg):
    return _predictor(lazy_regression=lazy_regression, **cfg) if not lazy_regression else\
         _lazy_predictor(lazy_regression=lazy_regression, **cfg)
    