from mimetypes import init
import torch
import math
import torch.distributed as dist
import pdb

from torch.nn import functional as F
from torch.nn.modules.loss import L1Loss

from mono3d.model.backbones.dla_dcn import IDAUp
from ...utils.comm import get_world_size

from ..anno_encoder import Anno_Encoder
from ..layers.utils import select_point_of_interest
from ..utils import Uncertainty_Reg_Loss, Laplace_Loss

from ..layers.focal_loss import *
from ..layers.iou_loss import *
from .depth_losses import *
from ..layers.utils import Converter_key2channel
from ..loss.score_loss import ScoreLoss
from mmcv.runner import BaseModule
from mono3d.structures.instance_pointcloud import InstancePoints
from mmdet3d.models.utils.utils_2d.boxes import Boxes


from typing import Dict, List, Tuple
from mmdet3d.models.utils.utils_2d.instances import Instances 
from torch.tensor import Tensor
from mmdet3d.models.roi_heads.poolers import ROIPooler


class AttentionHead(BaseModule):
	def __init__(self, 
        reg_heads, 
        reg_channels, 
		max_objs,
		center_sample: bool,
		mask_repair: bool, 
		pooler_cfg: dict, 
		heatmap_type,
		corner_depth_sp,
		loss_names,
		seg_uncertainty_range,
		init_loss_weight,
		seg_loss_type,
		init_cfg=None, 
		seg_feature_type='center_feature',
		instance_points_size_scale=1.,
		**kwargs
	):
		super().__init__(init_cfg=init_cfg)
		
		kwargs2 = locals().copy(); kwargs2.pop('self')
		kwargs2.update(kwargs)
		self.anno_encoder = Anno_Encoder(**kwargs2)
		self.key2channel = Converter_key2channel(
			keys=reg_heads, 
			channels=reg_channels
		)
		
		self.max_objs = max_objs
		self.center_sample = center_sample
		self.heatmap_type = heatmap_type
		self.corner_depth_sp = corner_depth_sp
		self.loss_keys = loss_names

		self.world_size = get_world_size()
		self.uncertainty_range = seg_uncertainty_range

		self.pooler = ROIPooler(**pooler_cfg)
		self.mask_size = pooler_cfg['output_size']
		self.mask_repair = mask_repair

		# loss functions
		if seg_loss_type == 'homo':
			self.seg_loss = CrossEntropyWithHomoUncertainty(alpha=0, )
		else: 
			self.seg_loss = CrossEntropyWithHeteroUncertainty(alpha=0)

		self.loss_weights = {}
		for key, weight in zip(loss_names, init_loss_weight): self.loss_weights[key] = weight

		# self.ida_up = IDAUp(64, [64, 128, 256, 512], [2 ** i for i in range(4)], )
		self.conv_pos = nn.Sequential(
			nn.Conv2d(2, 64, 1), 
			nn.ReLU(True), 
			nn.Conv2d(64, 64, 1),
			nn.ReLU(True),
		)
		self.conv = nn.Sequential(
			nn.Conv2d(64, 128, 1, bias=False),
			nn.GroupNorm(8, 128),
			nn.ReLU(True),
			nn.Conv2d(128, 64, 1, bias=False),
			nn.GroupNorm(8, 64),
			nn.ReLU(True)
		)
		self.conv_cls = nn.Sequential(
			nn.Conv2d(64, 1, 3, 1, 1),
		)
		if seg_loss_type != 'homo': 
			self.conv_uncertainty = nn.Sequential(
				nn.Conv2d(64, 1, 3, 1, 1),
			)
		self.seg_feature_type = seg_feature_type 
		self.instance_points_size_scale = instance_points_size_scale

		self.eps = 1e-5

	def pos_embed(self, device):
		yy, xx = torch.meshgrid(torch.arange(self.mask_size), torch.arange(self.mask_size),)
		pos = (torch.stack((yy, xx), axis=0).float() / self.mask_size) - 0.5
		pos = pos.to(device).unsqueeze(0)  # 1, 2, m, m
		pos = self.conv_pos(pos)
		return pos # 1, ch, mask_size, mask_size 

	def prepare_targets(self, inputs, outputs):
		img_metas = inputs['img_metas']
		batch = len(img_metas)
		gt_instances = inputs['gt_instances']
		gt_instance_ids = inputs['gt_instance_ids']  # (batch * H * W, 1)
		reg_indices = torch.where(gt_instance_ids >= 0)  
		gt_instance_ids_foreground = gt_instance_ids[reg_indices]  # (N, )
		down_ratio = self.anno_encoder.down_ratio

		# 1. get the representative points
		# targets_bbox_points = (torch.cat([i.gt_centers for i in gt_instances])[gt_instance_ids_foreground] / down_ratio).round().int()  # (N, 2)

		device = gt_instance_ids.device
		# the corresponding image_index for each object, used for finding pad_size, calib and so on
		batch_idxs = torch.cat([torch.full((len(instances),), i) for i, instances in enumerate(gt_instances)]).to(device).long() 
		batch_idxs = batch_idxs[gt_instance_ids_foreground]

		id_start = [0]
		for b in range(1, batch):
			id_start.append(id_start[-1] + len(gt_instances[b - 1]))
		
		idx_per_batch = [gt_instance_ids_foreground[batch_idxs == b] - id_start[b] for b in range(batch)]

		# fcos-style targets for 2D
		target_bboxes_2D = [i.gt_boxes[idx_per_batch[b]] for b, i in enumerate(gt_instances)]

		grids = inputs['grids'] / down_ratio  # M, 2 
		M = grids.shape[0]
		B = len(gt_instances)
		grids = grids.expand(B, M, 2).reshape(-1, 2)
		grids_selected = grids[reg_indices[0]]  # N, 2

		# targets for 3D
		target_bboxes_3D = [i.gt_boxes_3d[idx_per_batch[b]] for b, i in enumerate(gt_instances)]
		
		__ = compute_coord_2d(
			gt_instances, 
			[outputs['feature']], 
			self.pooler, 
			[i.gt_boxes for i in gt_instances], 
			inputs['img_metas'],
		)
		vu = [i.vu[idx_per_batch[b]] for b, i in enumerate(gt_instances)]

		# roi_imgs = ROIPooler(
        #     output_size=28,
        #     scales=(1.0,),
        #     sampling_ratio=0, 
        #     pooler_type='ROIAlignV2'
		# )([inputs['img']], target_bboxes_2D)
		# roi_imgs = self.pooler([inputs['img']], target_bboxes_2D)

		scaled_bboxes_3D = [target_bboxes_3D[b].clone() for b in range(batch)]
		if self.instance_points_size_scale != 1:
			for b in range(batch):
				scaled_bboxes_3D[b].tensor[:, 3:6] *= self.instance_points_size_scale 

		instance_points = [InstancePoints(inputs['points'][b], scaled_bboxes_3D[b]) for b in range(batch)]
		gt_masks = [
			instance_points[b].crop_and_resize(
				target_bboxes_2D[b].tensor, 
				self.mask_size, 
				vu[b], 
				img_metas[b]['calib']
			) for b, i in enumerate(gt_instances)
		]
		gt_masks = torch.cat([gt_masks[b][-1] for b in range(batch)], dim=0)
		if self.mask_repair: 
			gt_masks = mask_repair(gt_masks)
		return {
			'mask': gt_masks, 
			'reg_indices': reg_indices
		}

	def prepare_predictions(self, inputs, outputs, predictions, predictor, targets=None):
		img_metas = inputs['img_metas']
		pred_regression = predictions['reg']
		batch = pred_regression.shape[0]
		down_ratio = self.anno_encoder.down_ratio
		device = pred_regression.device

		if targets is not None: 
			reg_indices = targets['reg_indices']
		else: 
			reg_indices = outputs['reg_indices']

		if reg_indices[0].numel() == 0: 
			return {
				'mask': pred_regression.new_empty(0, 1, self.mask_size, self.mask_size)
			}

		# 1. get the representative points
		# targets_bbox_points = (torch.cat([i.gt_centers for i in gt_instances])[gt_instance_ids_foreground] / down_ratio).round().int()  # (N, 2)
		
		feature = outputs['feature']
		if self.seg_feature_type == 'center_feature':
			# fpn = outputs['fpn']
			# self.ida_up(fpn, 0, len(fpn))
			# feature_key = fpn[-1]
			query = select_point_of_interest(batch, reg_indices[0], feature).view(-1, feature.shape[1])[:, :, None, None]  # n, ch, 1, 1
			# roi_feature = self.pooler([feature_key], target_bboxes_2D)  # n, ch, m, m
			roi_feature = self.pos_embed(feature.device).expand((query.shape[0], 64, self.mask_size, self.mask_size))

			roi_feature = self.conv(query + roi_feature)
		elif self.seg_feature_type == 'roi_align': 
			gt_instances = inputs['gt_instances']
			gt_instance_ids = inputs['gt_instance_ids']
			gt_instance_ids_foreground = gt_instance_ids[reg_indices]
			batch_idxs = torch.cat([torch.full((len(instances),), i) for i, instances in enumerate(gt_instances)]).to(device).long()[gt_instance_ids_foreground]
			boxes = Boxes.cat([i.gt_boxes for i in gt_instances])[gt_instance_ids_foreground]
			boxes = [boxes[batch_idxs == b] for b in range(batch)]
			roi_feature = self.pooler(
				[feature], 
				boxes
			)

		mask = self.conv_cls(roi_feature).sigmoid().clamp(1e-4, 1-1e-4)

		ret = {
			'mask': mask 
		}

		if hasattr(self, 'conv_uncertainty'): 
			uncertainty = self.conv_uncertainty(roi_feature)
			uncertainty = uncertainty.clamp(*self.uncertainty_range)
			ret.update({'uncertainty': uncertainty})
		return ret 

	def forward(self, *args, **kwargs):
		return self.forward_train(*args, **kwargs)

	def forward_train(self, predictor, predictions, inputs, outputs):
		batch_size = len(inputs['img_metas'])
		
		targets = self.prepare_targets(inputs, outputs) 
		preds = self.prepare_predictions(inputs, outputs, predictions, predictor, targets)
		n = preds['mask'].shape[0]
		
		n_avg = preds['mask'].new_tensor(n)
		if dist.is_initialized() and dist.is_available():
			n_world = dist.get_world_size()
			dist.all_reduce(n_avg.div_(n_world))
		n_avg.clamp_(1.) 

		if 'uncertainty' not in preds: 
			seg_loss, _ = self.seg_loss(preds['mask'], targets['mask'])
		else: 
			seg_loss, _ = self.seg_loss(preds['mask'], preds['uncertainty'], targets['mask'])
		seg_loss = seg_loss / (n_avg * self.mask_size**2) * self.loss_weights['seg_loss']

		loss_dict = {
			'seg_loss': seg_loss, 
		}
		return loss_dict

	def forward_test(self, predictor, predictions, inputs, outputs):
		preds = self.prepare_predictions(inputs, outputs, predictions, predictor,)
		return preds 


class CrossEntropyWithHeteroUncertainty(BaseModule): 
	def __init__(self, alpha=2, beta=4, init_cfg=None):
		super().__init__(init_cfg)
		super(CrossEntropyWithHeteroUncertainty, self).__init__(init_cfg=init_cfg)
		self.alpha = alpha
		self.beta = beta

	def forward(self, prediction, uncertainty, target): 
		positive_index = target.eq(1).float()
		negative_index = (target.lt(1) & target.ge(0)).float()
		ignore_index = target.eq(-1).float() # ignored pixels

		negative_weights = torch.pow(1 - target, self.beta)
		loss = 0.

		positive_loss = torch.log(prediction) \
						* torch.pow(1 - prediction, self.alpha) * positive_index * torch.exp(-2 * uncertainty)
						
		negative_loss = torch.log(1 - prediction) \
						* torch.pow(prediction, self.alpha) * negative_weights * negative_index * torch.exp(-2 * uncertainty)

		num_positive = positive_index.float().sum()
		positive_loss = (-positive_loss.sum())
		negative_loss = (-negative_loss.sum())

		loss = negative_loss + positive_loss + uncertainty.mul(1 - ignore_index).sum()

		return loss, num_positive

	
class CrossEntropyWithHomoUncertainty(BaseModule):
	def __init__(self, alpha=2, beta=4, init_uncertainty=0.):
		super(CrossEntropyWithHomoUncertainty, self).__init__()
		self.alpha = alpha
		self.beta = beta
		self.uncertainty = nn.Parameter(torch.tensor(init_uncertainty))

	def forward(self, prediction, target):
		positive_index = target.eq(1).float()
		negative_index = (target.lt(1) & target.ge(0)).float()
		ignore_index = target.eq(-1).float() # ignored pixels

		negative_weights = torch.pow(1 - target, self.beta)
		loss = 0.
		uncertainty = self.uncertainty

		positive_loss = torch.log(prediction) \
						* torch.pow(1 - prediction, self.alpha) * positive_index * torch.exp(-2 * uncertainty)
						
		negative_loss = torch.log(1 - prediction) \
						* torch.pow(prediction, self.alpha) * negative_weights * negative_index * torch.exp(-2 * uncertainty)

		num_positive = positive_index.float().sum()
		positive_loss = (-positive_loss.sum())
		negative_loss = (-negative_loss.sum())

		loss = negative_loss + positive_loss + uncertainty.mul(1 - ignore_index).sum()

		return loss, num_positive


def compute_coord_2d(instances: List[Instances], features:List[Tensor], pooler:ROIPooler, boxes, img_metas):
    batch_size = features[0].shape[0]
    h, w, _ = img_metas[0]['pad_shape']  # all input images have the same shape
    vu = [
        (
            torch.stack(torch.meshgrid(
                torch.arange(tensor.shape[2]), 
                torch.arange(tensor.shape[3])
            ), axis=0).expand(batch_size, -1, -1, -1)
        ).to(tensor.device).float() + 0.5
        for tensor in features
    ]
    vu = [
        (
            lambda t, h1, w1: 
            t.div(t.new_tensor([h1, w1]).view(2, 1, 1))
                .mul(t.new_tensor([h, w]).view(2, 1, 1))
        )(t, t.shape[2], t.shape[3], )
        for b, t in enumerate(vu) 
    ]
    ret = pooler(vu, boxes)

    count = 0
    for instances_per_image in instances:
        count2 = count + len(instances_per_image)
        instances_per_image.vu = ret[count:count2]
        count = count2

    return ret 


def morphology_close(x, kernel_size, padding):
	x = F.max_pool2d(x, kernel_size, stride=1, padding=padding) 
	x = -F.max_pool2d(-x, kernel_size, stride=1, padding=padding)
	return x 


@torch.no_grad()
def mask_repair(gt_mask, kernel_size=3, padding=1):
	pos_mask = (gt_mask == 1).float() 
	neg_mask = (gt_mask == 0).float() 
	pos_mask = morphology_close(pos_mask, kernel_size, padding=padding) 
	neg_mask = morphology_close(neg_mask, kernel_size, padding=padding)
	assert pos_mask.shape == gt_mask.shape and neg_mask.shape == gt_mask.shape 
	keep = (pos_mask == neg_mask).float()
	pos_mask = pos_mask * (1 - keep)
	neg_mask = neg_mask * (1 - keep)
	ret = pos_mask + keep * gt_mask
	return ret 
