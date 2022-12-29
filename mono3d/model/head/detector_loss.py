import torch
import math
import torch.distributed as dist
import pdb

from torch.nn import functional as F
from torch.nn.modules.loss import L1Loss
from ...utils.comm import get_world_size

from ..anno_encoder import Anno_Encoder
from ..layers.utils import select_point_of_interest
from ..utils import Uncertainty_Reg_Loss, Laplace_Loss

from ..layers.focal_loss import *
from ..layers.iou_loss import *
from ..head.depth_losses import *
from ..layers.utils import Converter_key2channel
from ..loss.score_loss import ScoreLoss


def make_loss_evaluator(**cfg):
	loss_evaluator = Loss_Computation(**cfg)
	return loss_evaluator

class Loss_Computation():
	def __init__(self, 
        reg_heads, 
        reg_channels, 
		max_objs,
		center_sample: bool,
		pred_distribution: bool, 
		consistency_eval: bool,
		regression_area,
		heatmap_type,
		corner_depth_sp,
		loss_names,
		loss_types,
		dim_weight,
		uncertainty_range,
		loss_penalty_alpha,
		loss_beta,
		orientation,
		orientation_bin_size,
		truncation_offset_loss,
		init_loss_weight,
		uncertainty_weight,
		keypoint_xy_weight, 
		keypoint_norm_factor, 
		modify_invalid_keypoint_depth,
		corner_loss_depth,
		**kwargs
	):
		
		kwargs2 = locals().copy(); kwargs2.pop('self')
		kwargs2.update(kwargs)
		self.anno_encoder = Anno_Encoder(**kwargs2)
		self.key2channel = Converter_key2channel(
			keys=reg_heads, 
			channels=reg_channels
		)
		
		self.max_objs = max_objs
		self.center_sample = center_sample
		self.regress_area = regression_area
		self.heatmap_type = heatmap_type
		self.corner_depth_sp = corner_depth_sp
		self.loss_keys = loss_names

		self.world_size = get_world_size()
		self.dim_weight = torch.tensor(dim_weight)
		self.uncertainty_range = uncertainty_range

		# loss functions
		self.cls_loss_fnc = FocalLoss(loss_penalty_alpha, loss_beta) # penalty-reduced focal loss
		self.iou_loss = IOULoss(loss_type=loss_types[2]) # iou loss for 2D detection
		self.pred_distribution = pred_distribution
		self.consistency_eval = consistency_eval

		# depth loss
		if loss_types[3] == 'berhu': self.depth_loss = Berhu_Loss()
		elif loss_types[3] == 'inv_sig': self.depth_loss = Inverse_Sigmoid_Loss()
		elif loss_types[3] == 'log': self.depth_loss = Log_L1_Loss()
		elif loss_types[3] == 'L1': self.depth_loss = F.l1_loss
		else: raise ValueError

		# regular regression loss
		self.reg_loss = loss_types[1]
		self.reg_loss_fnc = F.l1_loss if loss_types[1] == 'L1' else F.smooth_l1_loss
		self.keypoint_loss_fnc = F.l1_loss

		# multi-bin loss setting for orientation estimation
		self.multibin = (orientation == 'multi-bin')
		self.orien_bin_size = orientation_bin_size
		self.trunc_offset_loss_type = truncation_offset_loss

		self.loss_weights = {}
		for key, weight in zip(loss_names, init_loss_weight): self.loss_weights[key] = weight

		# whether to compute corner loss
		self.compute_direct_depth_loss = 'depth_loss' in self.loss_keys
		self.compute_keypoint_depth_loss = 'keypoint_depth_loss' in self.loss_keys
		self.compute_weighted_depth_loss = 'weighted_avg_depth_loss' in self.loss_keys
		self.compute_corner_loss = 'corner_loss' in self.loss_keys
		self.separate_trunc_offset = 'trunc_offset_loss' in self.loss_keys
		
		self.pred_direct_depth = 'depth' in self.key2channel.keys
		self.depth_with_uncertainty = 'depth_uncertainty' in self.key2channel.keys
		self.compute_keypoint_corner = 'corner_offset' in self.key2channel.keys
		self.corner_with_uncertainty = 'corner_uncertainty' in self.key2channel.keys

		self.uncertainty_weight = uncertainty_weight # 1.0
		self.keypoint_xy_weights = keypoint_xy_weight # [1, 1]
		self.keypoint_norm_factor = keypoint_norm_factor # 1.0
		self.modify_invalid_keypoint_depths = modify_invalid_keypoint_depth

		# depth used to compute 8 corners
		self.corner_loss_depth = corner_loss_depth
		self.eps = 1e-5

	def prepare_predictions(self, inputs, outputs, predictions, predictor):
		img_metas = inputs['img_metas']
		pred_regression = predictions['reg']
		batch = pred_regression.shape[0]
		gt_instances = inputs['gt_instances']
		gt_instance_ids = inputs['gt_instance_ids']  # (B * H * W, 1)
		reg_indices = torch.where(gt_instance_ids >= 0)  
		gt_instance_ids_foreground = gt_instance_ids[reg_indices]  # (N, )
		down_ratio = self.anno_encoder.down_ratio

		# 1. get the representative points
		# targets_bbox_points = (torch.cat([i.gt_centers for i in gt_instances])[gt_instance_ids_foreground] / down_ratio).round().int()  # (N, 2)

		device = gt_instance_ids.device
		# the corresponding image_index for each object, used for finding pad_size, calib and so on
		batch_idxs = torch.cat([torch.full((len(instances),), i) for i, instances in enumerate(gt_instances)]).to(device).long() 
		batch_idxs = batch_idxs[gt_instance_ids_foreground]
		projection_matrix = torch.stack([torch.tensor(img_metas[b]['calib']['P2']) for b in range(batch)]).to(device)[batch_idxs].float()
		img_size_per_obj = torch.stack([torch.tensor(img_metas[b]['img_shape']) for b in range(batch)]).to(device)[batch_idxs]
		img_size_per_obj = torch.stack((img_size_per_obj[:, 1], img_size_per_obj[:, 0]), dim=-1)

		# 
		target_clses = torch.cat([i.gt_classes for i in gt_instances])[gt_instance_ids_foreground]

		# fcos-style targets for 2D
		target_bboxes_2D = torch.cat([i.gt_boxes.tensor for i in gt_instances])[gt_instance_ids_foreground] / down_ratio # N, 4
		target_bboxes_height = target_bboxes_2D[:, 3] - target_bboxes_2D[:, 1]
		target_bboxes_width = target_bboxes_2D[:, 2] - target_bboxes_2D[:, 0]

		grids = inputs['grids'] / down_ratio  # M, 2 
		M = grids.shape[0]
		B = len(gt_instances)
		grids = grids.expand(B, M, 2).reshape(-1, 2)
		grids_selected = grids[reg_indices[0]]  # N, 2

		target_regression_2D = torch.cat((grids_selected - target_bboxes_2D[:, :2], target_bboxes_2D[:, 2:] - grids_selected), dim=1)
		mask_regression_2D = (target_bboxes_height > 0) & (target_bboxes_width > 0)
		assert mask_regression_2D.all()

		# targets for 3D
		# target_clses = torch.cat([i.gt_classes for i in gt_instances])[gt_instance_ids_foreground]  # N,
		target_bboxes_3D = torch.cat([i.gt_boxes_3d.tensor for i in gt_instances])[gt_instance_ids_foreground]  # N, 7
		target_depths_3D = target_bboxes_3D[:, 2]  # N, 
		target_rotys_3D = target_bboxes_3D[:, 6]   # N,
		target_alphas_3D = torch.cat([i.gt_alpha for i in gt_instances])[gt_instance_ids_foreground].flatten().contiguous()  # N, 
		target_project_centers = torch.cat([i.gt_project_center for i in gt_instances])[gt_instance_ids_foreground] / down_ratio
		target_offset_3D = target_project_centers - grids_selected  # N, 2
		target_dimensions_3D = target_bboxes_3D[:, 3:6].contiguous()
		target_orientation_3D = torch.cat([i.gt_orientation for i in gt_instances])[gt_instance_ids_foreground]
		target_locations_3D = torch.cat([i.gt_locations for i in gt_instances])[gt_instance_ids_foreground]

		target_corners_3D = self.anno_encoder.encode_box3d(target_rotys_3D, target_dimensions_3D, target_locations_3D)  # (N, 8, 3)

		target_trunc_mask = torch.cat([i.gt_trunc_mask for i in gt_instances])[gt_instance_ids_foreground]
		obj_weights = 1.  # all objects are of the same weights (for now)

		# 2. extract corresponding predictions
		channel = pred_regression.shape[1]
		pred_regression_pois_3D = select_point_of_interest(batch, reg_indices[0], pred_regression).view(-1, channel)  # N, C
		
		# pred_regression_2D = F.relu(pred_regression_pois_3D[:, self.key2channel('2d_dim')])
		pred_regression_2D = pred_regression_pois_3D[:, self.key2channel('2d_dim')]
		pred_offset_3D = pred_regression_pois_3D[:, self.key2channel('3d_offset')]
		pred_dimensions_offsets_3D = pred_regression_pois_3D[:, self.key2channel('3d_dim')]
		pred_orientation_3D = torch.cat((pred_regression_pois_3D[:, self.key2channel('ori_cls')], 
									pred_regression_pois_3D[:, self.key2channel('ori_offset')]), dim=1)

		# decode the pred residual dimensions to real dimensions
		pred_dimensions_3D = self.anno_encoder.decode_dimension(target_clses, pred_dimensions_offsets_3D)

		# preparing outputs
		targets = { 'reg_2D': target_regression_2D, 'offset_3D': target_offset_3D, 'depth_3D': target_depths_3D, 'orien_3D': target_orientation_3D,
					'dims_3D': target_dimensions_3D, 'corners_3D': target_corners_3D, 'width_2D': target_bboxes_width, 'rotys_3D': target_rotys_3D,
					'cat_3D': target_bboxes_3D, 'trunc_mask_3D': target_trunc_mask, 'height_2D': target_bboxes_height,
					'coord': grids_selected,
					'img_size_per_obj': img_size_per_obj,
				}

		preds = {'reg_2D': pred_regression_2D, 'offset_3D': pred_offset_3D, 'orien_3D': pred_orientation_3D, 'dims_3D': pred_dimensions_3D,}

		if self.pred_distribution:
			pred_cov = pred_regression_pois_3D[:, self.key2channel('cov')].view(pred_regression_pois_3D.shape[0], 3, 3)
			pred_cov = torch.bmm(pred_cov.permute(0, 2, 1), pred_cov) + torch.eye(3, device=pred_cov.device) * 1e-3
			preds.update({'inv_covariance': pred_cov})

		reg_nums = {'reg_3D': target_bboxes_3D.new_tensor(target_bboxes_3D.shape[0])}
		weights = {'object_weights': obj_weights}

		# predict the depth with direct regression
		if self.pred_direct_depth:
			pred_depths_offset_3D = pred_regression_pois_3D[:, self.key2channel('depth')].squeeze(-1)
			pred_direct_depths_3D = self.anno_encoder.decode_depth(pred_depths_offset_3D)
			preds['depth_3D'] = pred_direct_depths_3D

		# predict the uncertainty of depth regression
		if self.depth_with_uncertainty:
			preds['depth_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('depth_uncertainty')].squeeze(-1)
			
			if self.uncertainty_range is not None:
				preds['depth_uncertainty'] = torch.clamp(preds['depth_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])

			# else:
			# 	print('depth_uncertainty: {:.2f} +/- {:.2f}'.format(
			# 		preds['depth_uncertainty'].mean().item(), preds['depth_uncertainty'].std().item()))

		# predict the keypoints
		if self.compute_keypoint_corner:
			# targets for keypoints
			target_corner_keypoints = torch.cat([i.gt_keypoints for i in gt_instances])[gt_instance_ids_foreground]
			targets['keypoints'] = target_corner_keypoints[..., :2] / down_ratio - grids_selected[:, None, :]
			targets['keypoints_mask'] = target_corner_keypoints[..., -1]
			reg_nums['keypoints'] = targets['keypoints_mask'].sum()

			# mask for whether depth should be computed from certain group of keypoints
			target_corner_depth_mask = torch.cat([i.gt_keypoints_depth_mask for i in gt_instances])[gt_instance_ids_foreground]
			targets['keypoints_depth_mask'] = target_corner_depth_mask

			# predictions for keypoints
			pred_keypoints_3D = pred_regression_pois_3D[:, self.key2channel('corner_offset')]
			pred_keypoints_3D = pred_keypoints_3D.view(pred_keypoints_3D.shape[0], 10, 2)
			pred_keypoints_depths_3D = self.anno_encoder.decode_depth_from_keypoints_batch(
				inputs['img_metas'],
				pred_keypoints_3D, pred_dimensions_3D,
				batch_idxs
			)

			preds['keypoints'] = pred_keypoints_3D			
			preds['keypoints_depths'] = pred_keypoints_depths_3D

		# predict the uncertainties of the solved depths from groups of keypoints
		if self.corner_with_uncertainty:
			preds['corner_offset_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('corner_uncertainty')]

			if self.uncertainty_range is not None:
				preds['corner_offset_uncertainty'] = torch.clamp(preds['corner_offset_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])

			# else:
			# 	print('keypoint depth uncertainty: {:.2f} +/- {:.2f}'.format(
			# 		preds['corner_offset_uncertainty'].mean().item(), preds['corner_offset_uncertainty'].std().item()))

		# compute the corners of the predicted 3D bounding boxes for the corner loss
		if self.corner_loss_depth == 'direct':
			pred_corner_depth_3D = pred_direct_depths_3D

		elif self.corner_loss_depth == 'keypoint_mean':
			pred_corner_depth_3D = preds['keypoints_depths'].mean(dim=1)
		
		else:
			assert self.corner_loss_depth in ['soft_combine', 'hard_combine']
			# make sure all depths and their uncertainties are predicted
			pred_combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(-1), preds['corner_offset_uncertainty']), dim=1).exp()
			pred_combined_depths = torch.cat((pred_direct_depths_3D.unsqueeze(-1), preds['keypoints_depths']), dim=1)
			
			if self.corner_loss_depth == 'soft_combine':
				pred_uncertainty_weights = 1 / pred_combined_uncertainty
				pred_uncertainty_weights = pred_uncertainty_weights / pred_uncertainty_weights.sum(dim=1, keepdim=True)
				pred_corner_depth_3D = torch.sum(pred_combined_depths * pred_uncertainty_weights, dim=1)
				preds['weighted_depths'] = pred_corner_depth_3D
			
			elif self.corner_loss_depth == 'hard_combine':
				pred_corner_depth_3D = pred_combined_depths[torch.arange(pred_combined_depths.shape[0]), pred_combined_uncertainty.argmin(dim=1)]

		# compute the corners
		pred_locations_3D = self.anno_encoder.decode_location_flatten(inputs['img_metas'], grids_selected, pred_offset_3D, pred_corner_depth_3D, 
										batch_idxs)
		# decode rotys and alphas
		pred_rotys_3D, _ = self.anno_encoder.decode_axes_orientation(pred_orientation_3D, pred_locations_3D)
		# encode corners
		pred_corners_3D = self.anno_encoder.encode_box3d(pred_rotys_3D, pred_dimensions_3D, pred_locations_3D)
		# concatenate all predictions
		pred_bboxes_3D = torch.cat((pred_locations_3D, pred_dimensions_3D, pred_rotys_3D[:, None]), dim=1)

		if self.consistency_eval:
			pred_corners_proj, pred_corners_proj_mask = self.anno_encoder.decode_corners_proj(
				pred_rotys_3D, pred_dimensions_3D, 
				pred_locations_3D, img_size_per_obj,
				projection_matrix,
			)
			preds.update({
				'corners_proj': pred_corners_proj, 
				'corners_proj_mask': pred_corners_proj_mask,
			})

		preds.update({'corners_3D': pred_corners_3D, 'rotys_3D': pred_rotys_3D, 'cat_3D': pred_bboxes_3D})

		return targets, preds, reg_nums, weights

	def __call__(self, predictor, predictions, inputs, outputs):
		batch_size = len(inputs['img_metas'])
		targets_heatmap = inputs['gt_heatmap']

		pred_heatmap = predictions['cls']
		pred_targets, preds, reg_nums, weights = self.prepare_predictions(inputs, outputs, predictions, predictor)

		# heatmap loss
		if self.heatmap_type == 'centernet':
			hm_loss, num_hm_pos = self.cls_loss_fnc(pred_heatmap, targets_heatmap)

		else: raise ValueError

		n_kpts = reg_nums['keypoints']
		num_reg_2D = num_reg_3D = num_reg_obj = reg_nums['reg_3D']

		if self.compute_keypoint_depth_loss:
			pred_keypoints_depth, keypoints_depth_mask = preds['keypoints_depths'], pred_targets['keypoints_depth_mask'].bool()
			n_kpts_depth_valid = keypoints_depth_mask.sum()
			n_kpts_depth_invalid = (~keypoints_depth_mask).sum()
		if dist.is_initialized() and dist.is_available():
			n_world = dist.get_world_size()
			dist.all_reduce(num_hm_pos.div_(n_world))
			dist.all_reduce(num_reg_2D.div_(n_world))
			dist.all_reduce(num_reg_3D.div_(n_world))
			if self.compute_keypoint_depth_loss:
				dist.all_reduce(n_kpts_depth_invalid.div_(n_world))
				dist.all_reduce(n_kpts_depth_valid.div_(n_world))
		num_hm_pos.clamp_(1.)
		n_kpts.clamp_(1.)
		num_reg_2D.clamp_(1.)
		num_reg_3D.clamp_(1.)
		if self.compute_keypoint_depth_loss:
			n_kpts_depth_valid.clamp_(1.)
			n_kpts_depth_invalid.clamp_(1.)
		
		hm_loss = self.loss_weights['hm_loss'] * hm_loss / num_hm_pos

		trunc_mask = pred_targets['trunc_mask_3D'].bool()
		num_trunc = trunc_mask.sum()
		num_nontrunc = num_reg_obj - num_trunc

		# IoU loss for 2D detection
		if len(preds['reg_2D']) > 0:
			bbox_loss, iou_2D = self.iou_loss(preds['reg_2D'], pred_targets['reg_2D'])
			bbox_l1_loss = self.reg_loss_fnc(preds['reg_2D'], pred_targets['reg_2D'], reduction='none').sum().div(num_reg_2D) * self.loss_weights['bbox_l1_loss']
			bbox_iou_loss = bbox_loss.sum().div(num_reg_2D) * self.loss_weights['bbox_iou_loss']
			iou_2D = iou_2D.sum() / num_reg_2D
			depth_MAE = (preds['depth_3D'] - pred_targets['depth_3D']).abs() / pred_targets['depth_3D'] \
				if self.pred_direct_depth else None 
		else:
			bbox_l1_loss = bbox_iou_loss = \
				iou_2D = pred_heatmap.new_tensor(0.)
			depth_MAE = torch.zeros_like(pred_targets['depth_3D'])\
				if self.pred_direct_depth else None 

		if len(preds['depth_3D']) > 0:
			# direct depth loss
			if self.compute_direct_depth_loss:
				depth_3D_loss = self.loss_weights['depth_loss'] * self.depth_loss(preds['depth_3D'], pred_targets['depth_3D'], reduction='none')
				real_depth_3D_loss = depth_3D_loss.detach().mean()
				
				if self.depth_with_uncertainty:
					depth_3D_loss = depth_3D_loss * torch.exp(- preds['depth_uncertainty']) + \
							preds['depth_uncertainty'] * self.loss_weights['depth_loss']
				
				depth_3D_loss = depth_3D_loss.mean()
			
			# offset_3D loss
			offset_3D_loss = self.reg_loss_fnc(preds['offset_3D'], pred_targets['offset_3D'], reduction='none').sum(dim=1)
			
			# use different loss functions for inside and outside objects
			if self.separate_trunc_offset:
				if self.trunc_offset_loss_type == 'L1':
					trunc_offset_loss = offset_3D_loss[trunc_mask]
				
				elif self.trunc_offset_loss_type == 'log':
					trunc_offset_loss = torch.log(1 + offset_3D_loss[trunc_mask])

				trunc_offset_loss = self.loss_weights['trunc_offset_loss'] * trunc_offset_loss.sum() / torch.clamp(trunc_mask.sum(), min=1)
				offset_3D_loss = self.loss_weights['offset_loss'] * offset_3D_loss[~trunc_mask].mean() \
					if (~trunc_mask).sum() > 0 else hm_loss.new_tensor(0.)
			else:
				offset_3D_loss = self.loss_weights['offset_loss'] * offset_3D_loss.mean()

			# orientation loss
			if self.multibin:
				orien_3D_loss = self.loss_weights['orien_loss'] * \
								Real_MultiBin_loss(preds['orien_3D'], pred_targets['orien_3D'], num_bin=self.orien_bin_size)

			# dimension loss
			dims_3D_loss = self.reg_loss_fnc(preds['dims_3D'], pred_targets['dims_3D'], reduction='none') * self.dim_weight.type_as(preds['dims_3D'])
			dims_3D_loss = self.loss_weights['dims_loss'] * dims_3D_loss.sum(dim=1).mean()

			with torch.no_grad(): 
				IoU_3D = get_iou_3d(preds['corners_3D'], pred_targets['corners_3D'])
				pred_IoU_3D = IoU_3D.sum() / num_reg_3D

			# corner loss
			if self.compute_corner_loss:
				# N x 8 x 3
				corner_3D_loss = self.loss_weights['corner_loss'] * \
							self.reg_loss_fnc(preds['corners_3D'], pred_targets['corners_3D'], reduction='none').sum(dim=2).mean()

			if self.compute_keypoint_corner:
				# N x K x 3
				keypoint_loss = self.loss_weights['keypoint_loss'] * self.keypoint_loss_fnc(preds['keypoints'],
								pred_targets['keypoints'], reduction='none').sum(dim=2) * pred_targets['keypoints_mask']
				
				keypoint_loss = keypoint_loss.sum() / n_kpts

				if self.compute_keypoint_depth_loss:
					target_keypoints_depth = pred_targets['depth_3D'].unsqueeze(-1).repeat(1, 3)
					
					valid_pred_keypoints_depth = pred_keypoints_depth[keypoints_depth_mask]
					invalid_pred_keypoints_depth = pred_keypoints_depth[~keypoints_depth_mask].detach()
					
					# valid and non-valid
					valid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(valid_pred_keypoints_depth, 
															target_keypoints_depth[keypoints_depth_mask], reduction='none')
					
					invalid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(invalid_pred_keypoints_depth, 
															target_keypoints_depth[~keypoints_depth_mask], reduction='none')
					
					# for logging
					log_valid_keypoint_depth_loss = valid_keypoint_depth_loss.detach().mean()

					if self.corner_with_uncertainty:
						# center depth, corner 0246 depth, corner 1357 depth
						pred_keypoint_depth_uncertainty = preds['corner_offset_uncertainty']

						valid_uncertainty = pred_keypoint_depth_uncertainty[keypoints_depth_mask]
						invalid_uncertainty = pred_keypoint_depth_uncertainty[~keypoints_depth_mask]

						valid_keypoint_depth_loss = valid_keypoint_depth_loss * torch.exp(- valid_uncertainty) + \
												self.loss_weights['keypoint_depth_loss'] * valid_uncertainty

						invalid_keypoint_depth_loss = invalid_keypoint_depth_loss * torch.exp(- invalid_uncertainty)
					
					# average
					valid_keypoint_depth_loss = valid_keypoint_depth_loss.sum() / n_kpts_depth_valid
					invalid_keypoint_depth_loss = invalid_keypoint_depth_loss.sum() / n_kpts_depth_invalid

					# the gradients of invalid depths are not back-propagated
					if self.modify_invalid_keypoint_depths:
						keypoint_depth_loss = valid_keypoint_depth_loss + invalid_keypoint_depth_loss
					else:
						keypoint_depth_loss = valid_keypoint_depth_loss
				
				# compute the average error for each method of depth estimation
				keypoint_MAE = (preds['keypoints_depths'] - pred_targets['depth_3D'].unsqueeze(-1)).abs() \
									/ pred_targets['depth_3D'].unsqueeze(-1)
				
				center_MAE = keypoint_MAE[:, 0].mean()
				keypoint_02_MAE = keypoint_MAE[:, 1].mean()
				keypoint_13_MAE = keypoint_MAE[:, 2].mean()

				if self.corner_with_uncertainty:
					if self.pred_direct_depth and self.depth_with_uncertainty:
						combined_depth = torch.cat((preds['depth_3D'].unsqueeze(1), preds['keypoints_depths']), dim=1)
						combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(1), preds['corner_offset_uncertainty']), dim=1).exp()
						combined_MAE = torch.cat((depth_MAE.unsqueeze(1), keypoint_MAE), dim=1)
					else:
						combined_depth = preds['keypoints_depths']
						combined_uncertainty = preds['corner_offset_uncertainty'].exp()
						combined_MAE = keypoint_MAE

					# the oracle MAE
					lower_MAE = torch.min(combined_MAE, dim=1)[0]
					# the hard ensemble
					hard_MAE = combined_MAE[torch.arange(combined_MAE.shape[0]), combined_uncertainty.argmin(dim=1)]
					# the soft ensemble
					combined_weights = 1 / combined_uncertainty
					combined_weights = combined_weights / combined_weights.sum(dim=1, keepdim=True)
					soft_depths = torch.sum(combined_depth * combined_weights, dim=1)
					soft_MAE = (soft_depths - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']
					# the average ensemble
					mean_depths = combined_depth.mean(dim=1)
					mean_MAE = (mean_depths - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']

					# average
					lower_MAE, hard_MAE, soft_MAE, mean_MAE = lower_MAE.mean(), hard_MAE.mean(), soft_MAE.mean(), mean_MAE.mean()
				
					if self.compute_weighted_depth_loss:
						soft_depth_loss = self.loss_weights['weighted_avg_depth_loss'] * \
										self.reg_loss_fnc(soft_depths, pred_targets['depth_3D'], reduction='mean')

			if self.pred_distribution:
				location_3D = preds['cat_3D'][:, :3, None]  # n, 3, 1
				target_location_3D = pred_targets['cat_3D'][:, :3, None]  # n, 3, 1
				location_error = (target_location_3D - location_3D).detach()  # n, 3, 1
				pred_inv_cov = preds['inv_covariance']  # n, 3, 3
				cov_loss = torch.bmm(location_error.permute(0, 2, 1), torch.bmm(pred_inv_cov, location_error)) - torch.det(pred_inv_cov).log().unsqueeze(-1).unsqueeze(-1)
				cov_loss = cov_loss.sum() / num_reg_3D * self.loss_weights['cov']
			
			if self.consistency_eval:
				consistency_loss = self.consistency_error(preds['corners_proj'], preds['corners_proj_mask'], preds['keypoints'], pred_targets['img_size_per_obj'], pred_targets['coord'], pred_targets['width_2D'],
				pred_targets['height_2D']) * self.loss_weights['consistency_loss']
			depth_MAE = depth_MAE.mean() if depth_MAE is not None else None
		else:
			dims_3D_loss = orien_3D_loss = pred_IoU_3D = offset_3D_loss =              \
			trunc_offset_loss = corner_3D_loss = depth_3D_loss =                       \
			real_depth_3D_loss = depth_MAE = keypoint_loss =                           \
			center_MAE = keypoint_02_MAE = keypoint_13_MAE = lower_MAE =               \
			hard_MAE = soft_MAE = mean_MAE = keypoint_depth_loss =                     \
			log_valid_keypoint_depth_loss = soft_depth_loss =                          \
			cov_loss = 											   \
			consistency_loss = \
				pred_heatmap.mul(0).sum()

		loss_dict = {
			'hm_loss':  hm_loss,
			'bbox_iou_loss': bbox_iou_loss,
			'bbox_l1_loss': bbox_l1_loss,
			'dims_loss': dims_3D_loss,
			'orien_loss': orien_3D_loss,
		}

		if self.pred_distribution: loss_dict.update({'cov_loss': cov_loss})

		log_loss_dict = {
			'2D_IoU': iou_2D,
			'3D_IoU': pred_IoU_3D,
		}

		MAE_dict = {}

		if self.separate_trunc_offset:
			loss_dict['offset_loss'] = offset_3D_loss
			loss_dict['trunc_offset_loss'] = trunc_offset_loss
		else:
			loss_dict['offset_loss'] = offset_3D_loss

		if self.compute_corner_loss:
			loss_dict['corner_loss'] = corner_3D_loss

		if self.pred_direct_depth:
			loss_dict['depth_loss'] = depth_3D_loss
			log_loss_dict['depth_loss'] = real_depth_3D_loss
			if depth_MAE is not None: MAE_dict['depth_MAE'] = depth_MAE

		if self.compute_keypoint_corner:
			loss_dict['keypoint_loss'] = keypoint_loss

			MAE_dict.update({
				'center_MAE': center_MAE,
				'02_MAE': keypoint_02_MAE,
				'13_MAE': keypoint_13_MAE,
			})

			if self.corner_with_uncertainty:
				MAE_dict.update({
					'lower_MAE': lower_MAE,
					'hard_MAE': hard_MAE,
					'soft_MAE': soft_MAE,
					'mean_MAE': mean_MAE,
				})

		if self.compute_keypoint_depth_loss:
			loss_dict['keypoint_depth_loss'] = keypoint_depth_loss
			log_loss_dict['keypoint_depth_loss'] = log_valid_keypoint_depth_loss

		if self.compute_weighted_depth_loss:
			loss_dict['weighted_avg_depth_loss'] = soft_depth_loss
		
		if self.consistency_eval:
			loss_dict['consistency_loss'] = consistency_loss

		# loss_dict ===> log_loss_dict
		for key, value in loss_dict.items():
			if key not in log_loss_dict:
				log_loss_dict[key] = value

		# stop when the loss has NaN or Inf
		for v in loss_dict.values():
			if torch.isnan(v).sum() > 0:
				pdb.set_trace()
			if torch.isinf(v).sum() > 0:
				pdb.set_trace()

		log_loss_dict.update(MAE_dict)

		return loss_dict, log_loss_dict

	def consistency_error(self, pred_corners_proj, pred_corners_proj_mask,  pred_keypoint_offset, img_size_per_obj, pred_bbox_points, target_height, target_width, eps=1e-8, alpha=1000):
		pred_keypoint_offset = pred_keypoint_offset\
			.add(pred_bbox_points.unsqueeze(1))\
			.mul(self.anno_encoder.down_ratio)
		pred_corners_proj = pred_corners_proj
		mask = (
			pred_corners_proj_mask 
			& (pred_keypoint_offset[..., 0] >= 0)
			& (pred_keypoint_offset[..., 1] >= 0)
			& (pred_keypoint_offset[..., 0] < img_size_per_obj[:, None, 0])
			& (pred_keypoint_offset[..., 1] < img_size_per_obj[:, None, 1])
		)
		error = (
			pred_keypoint_offset - pred_corners_proj
		).abs().sum(-1).mul(mask).mean(-1)  # n, )
		# box_area = target_height * target_width
		# error = error.div(box_area.clamp(eps)) * alpha
		return error.mean()

def Real_MultiBin_loss(vector_ori, gt_ori, num_bin=4):
	gt_ori = gt_ori.view(-1, gt_ori.shape[-1]) # bin1 cls, bin1 offset, bin2 cls, bin2 offst

	cls_losses = 0
	reg_losses = 0
	reg_cnt = 0
	for i in range(num_bin):
		# bin cls loss
		cls_ce_loss = F.cross_entropy(vector_ori[:, (i * 2) : (i * 2 + 2)], gt_ori[:, i].long(), reduction='none')
		# regression loss
		valid_mask_i = (gt_ori[:, i] == 1)
		cls_losses += cls_ce_loss.mean()
		if valid_mask_i.sum() > 0:
			s = num_bin * 2 + i * 2
			e = s + 2
			pred_offset = F.normalize(vector_ori[valid_mask_i, s : e])
			reg_loss = F.l1_loss(pred_offset[:, 0], torch.sin(gt_ori[valid_mask_i, num_bin + i]), reduction='none') + \
						F.l1_loss(pred_offset[:, 1], torch.cos(gt_ori[valid_mask_i, num_bin + i]), reduction='none')

			reg_losses += reg_loss.sum()
			reg_cnt += valid_mask_i.sum()

	return cls_losses / num_bin + reg_losses / reg_cnt
