import torch
import pdb
import math
import time 

from torch import nn
from shapely.geometry import Polygon
from torch.nn import functional as F
from mmdet3d.utils.misc import pad_ones

from ...model.anno_encoder import Anno_Encoder, get_covariance_from_uncertainty
from ...model.layers.utils import (
    nms_hm,
    nms_hm_dilate,
    select_topk,
    select_point_of_interest,
)

from ...model.layers.utils import Converter_key2channel

from typing import List 


def make_post_processor(reg_heads, reg_channels, **cfg):
    anno_encoder = Anno_Encoder(**cfg)
    key2channel = Converter_key2channel(keys=reg_heads, channels=reg_channels)
    postprocessor = PostProcessor(anno_encoder=anno_encoder, key2channel=key2channel, **cfg)
    
    return postprocessor

class PostProcessor(nn.Module):
    def __init__(self, 
        input_width,
        input_height,
        down_ratio,
        anno_encoder, key2channel,
        depth_output,
        test_cfg,
        num_classes,
        quantization_method,
        pred_distribution,
        pred_features,
        consistency_eval,
        cls_blacklist,
        **kwargs
    ):
        
        super(PostProcessor, self).__init__()

        self.anno_encoder = anno_encoder
        self.key2channel = key2channel

        self.det_threshold = test_cfg.score_threshold
        self.max_detection = test_cfg.max_per_img	
        self.eval_dis_iou = test_cfg.eval_dis_ious
        self.eval_depth = test_cfg.eval_depth
        
        self.output_width = input_width // down_ratio
        self.output_height = input_height // down_ratio
        self.output_depth = depth_output
        self.pred_2d = test_cfg.pred_2d

        self.pred_direct_depth = 'depth' in self.key2channel.keys
        self.depth_with_uncertainty = 'depth_uncertainty' in self.key2channel.keys
        self.regress_keypoints = 'corner_offset' in self.key2channel.keys
        self.keypoint_depth_with_uncertainty = 'corner_uncertainty' in self.key2channel.keys

        # use uncertainty to guide the confidence
        self.uncertainty_as_conf = test_cfg.uncertainty_as_confidence
        self.quantization_method = quantization_method
        self.consistency_eval = consistency_eval
        self.pred_distribution = pred_distribution
        self.pred_features = pred_features
        self.num_classes = num_classes
        self.cls_blacklist = cls_blacklist
        
    def forward(self, predictions, predictor, img_metas: List[dict], features=None, test=False, refine_module=None, depth_disturb=False):
        pred_heatmap, pred_regression = predictions['cls'], predictions['reg']
        batch = pred_heatmap.shape[0]
        img_size = [img_metas[b]['img_shape'] for b in range(batch)]
        img_size = torch.tensor([[i[1], i[0]] for i in img_size])

        target_varibales = {}

        # evaluate the disentangling IoU for each components in (location, dimension, orientation)
        dis_ious = self.evaluate_3D_detection(target_varibales, pred_regression) if self.eval_dis_iou else None

        # evaluate the accuracy of predicted depths
        depth_errors = self.evaluate_3D_depths(target_varibales, pred_regression) if self.eval_depth else None

        pred_heatmap = pred_heatmap[:, :self.num_classes] #* pred_heatmap[:, [self.num_classes + 1]]

        # max-pooling as nms for heat-map
        heatmap = nms_hm(pred_heatmap) 
        visualize_preds = {'heat_map': pred_heatmap.clone()}

        # select top-k of the predicted heatmap
        scores_cls, indexs, clses, ys, xs = select_topk(heatmap, K=self.max_detection)
        batch_idxs = torch.arange(0, batch).to(xs.device).unsqueeze(-1).expand(batch, self.max_detection).flatten()
        img_size_per_obj = img_size[batch_idxs].to(xs.device)
        projection_matrix_per_obj = torch.stack([torch.tensor(img_metas[b]['calib']['P2']).float() for b in range(batch)])[batch_idxs].to(xs.device)

        pred_bbox_points = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1) + (0.5 if self.quantization_method == 'floor' else 0)
        pred_regression_pois = select_point_of_interest(batch, indexs, pred_regression).view(-1, pred_regression.shape[1])

        # thresholding with score
        scores_cls = scores_cls.view(-1)
        valid_mask = scores_cls >= self.det_threshold
        if self.cls_blacklist is not None:
            valid_mask = (valid_mask & (clses[:, :, None] != clses.new_tensor(self.cls_blacklist)[None, None, :]).all(dim=-1).flatten())

        # no valid predictions
        if valid_mask.sum() == 0:
            result = scores_cls.new_zeros(0, 14)
            visualize_preds['keypoints'] = scores_cls.new_zeros(0, 10, 2)
            visualize_preds['proj_center'] = scores_cls.new_zeros(0, 2)
            eval_utils = {'vis_scores': scores_cls.new_zeros(0),
                    'uncertainty_conf': scores_cls.new_zeros(0), 'estimated_depth_error': scores_cls.new_zeros(0),
                    'batch_indices': torch.empty(0,).long(),
                    'reg_indices': torch.empty(1, 0).long()}
            if dis_ious is not None: 
                eval_utils['dis_ious'] = dis_ious
            if depth_errors is not None: 
                eval_utils['depth_errors'] = depth_errors
            if self.pred_distribution:
                eval_utils['precision'] = scores_cls.new_empty(0, 3, 3)
            if self.pred_features:
                eval_utils['features'] = scores_cls.new_empty(0, features.shape[1])
            eval_utils['scores_cls'] = scores_cls.new_empty(0,)
            return result, eval_utils, visualize_preds

        scores_cls = scores_cls[valid_mask]
        
        indexs = indexs[:, valid_mask]
        clses = clses.view(-1)[valid_mask]
        pred_bbox_points = pred_bbox_points[valid_mask]
        pred_regression_pois = pred_regression_pois[valid_mask]
        batch_idxs = batch_idxs[valid_mask]
        img_size_per_obj = img_size_per_obj[valid_mask]
        projection_matrix_per_obj = projection_matrix_per_obj[valid_mask]

        pred_2d_reg = F.relu(pred_regression_pois[:, self.key2channel('2d_dim')])
        pred_offset_3D = pred_regression_pois[:, self.key2channel('3d_offset')]
        pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
        pred_orientation = torch.cat((pred_regression_pois[:, self.key2channel('ori_cls')], pred_regression_pois[:, self.key2channel('ori_offset')]), dim=1)
        visualize_preds['proj_center'] = pred_bbox_points + pred_offset_3D

        pred_box2d = self.anno_encoder.decode_box2d_fcos(pred_bbox_points, pred_2d_reg, img_size_per_obj)
        pred_dimensions = self.anno_encoder.decode_dimension(clses, pred_dimensions_offsets)

        if self.pred_direct_depth:
            pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')].squeeze(-1)
            pred_direct_depths = self.anno_encoder.decode_depth(pred_depths_offset)

        if self.depth_with_uncertainty:
            pred_direct_uncertainty = pred_regression_pois[:, self.key2channel('depth_uncertainty')].exp()
            visualize_preds['depth_uncertainty'] = pred_regression[:, self.key2channel('depth_uncertainty'), ...].squeeze(1)

        if self.regress_keypoints:
            pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')]
            pred_keypoint_offset = pred_keypoint_offset.view(-1, 10, 2)
            # solve depth from estimated key-points
            pred_keypoints_depths = self.anno_encoder.decode_depth_from_keypoints_batch(img_metas, pred_keypoint_offset, pred_dimensions, batch_idxs)
            visualize_preds['keypoints'] = pred_keypoint_offset

        if self.keypoint_depth_with_uncertainty:
            pred_keypoint_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()

        estimated_depth_error = None

        if self.output_depth == 'direct':
            pred_depths = pred_direct_depths

            if self.depth_with_uncertainty: estimated_depth_error = pred_direct_uncertainty.squeeze(dim=1)
        
        elif self.output_depth.find('keypoints') >= 0:
            if self.output_depth == 'keypoints_avg':
                pred_depths = pred_keypoints_depths.mean(dim=1)
                if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty.mean(dim=1)

            elif self.output_depth == 'keypoints_center':
                pred_depths = pred_keypoints_depths[:, 0]
                if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty[:, 0]

            elif self.output_depth == 'keypoints_02':
                pred_depths = pred_keypoints_depths[:, 1]
                if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty[:, 1]

            elif self.output_depth == 'keypoints_13':
                pred_depths = pred_keypoints_depths[:, 2]
                if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty[:, 2]

            else:
                raise ValueError

        # hard ensemble, soft ensemble and simple average
        elif self.output_depth in ['hard', 'soft', 'mean', 'oracle']:
            if self.pred_direct_depth and self.depth_with_uncertainty:
                pred_combined_depths = torch.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), dim=1)
                pred_combined_uncertainty = torch.cat((pred_direct_uncertainty, pred_keypoint_uncertainty), dim=1)
            else:
                pred_combined_depths = pred_keypoints_depths.clone()
                pred_combined_uncertainty = pred_keypoint_uncertainty.clone()
            
            depth_weights = 1 / pred_combined_uncertainty
            visualize_preds['min_uncertainty'] = depth_weights.argmax(dim=1)

            if self.output_depth == 'hard':
                pred_depths = pred_combined_depths[torch.arange(pred_combined_depths.shape[0]), depth_weights.argmax(dim=1)]

                # the uncertainty after combination				
                estimated_depth_error = pred_combined_uncertainty.min(dim=1).values

            elif self.output_depth == 'soft':
                depth_weights = depth_weights / depth_weights.sum(dim=1, keepdim=True)
                pred_depths = torch.sum(pred_combined_depths * depth_weights, dim=1)

                # the uncertainty after combination
                estimated_depth_error = torch.sum(depth_weights * pred_combined_uncertainty, dim=1)
                
            elif self.output_depth == 'mean':
                pred_depths = pred_combined_depths.mean(dim=1)

                # the uncertainty after combination
                estimated_depth_error = pred_combined_uncertainty.mean(dim=1)

            # the best estimator is always selected
            elif self.output_depth == 'oracle':
                pred_depths, estimated_depth_error = self.get_oracle_depths(pred_box2d, clses, pred_combined_depths, 
                                                                pred_combined_uncertainty, targets[0])

        if depth_disturb:
            sigma = torch.exp(pred_depths / 40) - 1 + 1e-2
            pred_depths = pred_depths + torch.randn(pred_depths.shape[0]).to(pred_depths.device) * sigma

        pred_locations = self.anno_encoder.decode_location_flatten(img_metas, pred_bbox_points, pred_offset_3D, pred_depths, batch_idxs)
        pred_rotys, pred_alphas = self.anno_encoder.decode_axes_orientation(pred_orientation, pred_locations)

        # pred_locations[:, 1] += pred_dimensions[:, 1] / 2
        clses = clses.view(-1, 1)
        pred_alphas = pred_alphas.view(-1, 1)
        pred_rotys = pred_rotys.view(-1, 1)
        scores_cls = scores_cls.view(-1, 1)

        # the uncertainty of depth estimation can reflect the confidence for 3D object detection
        vis_scores = scores_cls.clone()
        if self.uncertainty_as_conf and estimated_depth_error is not None:
            uncertainty_limit = 1.
            uncertainty_conf = (uncertainty_limit - torch.clamp(estimated_depth_error, min=0.01, max=uncertainty_limit)) / uncertainty_limit
            # uncertainty_conf = torch.exp(1 - torch.clamp(estimated_depth_error, min=0.01))
            scores = scores_cls * uncertainty_conf.view(-1, 1)
        else:
            uncertainty_conf, estimated_depth_error = None, None

        if self.pred_distribution:
            # A = pred_regression_pois[:, self.key2channel('cov')].view(pred_regression_pois.shape[0], 3, 3)
            # precision = torch.bmm(A.permute(0, 2, 1), A) + torch.eye(3, device=A.device) * 1e-3
            precision = self.anno_encoder.decode_precision_matrix(img_metas, pred_bbox_points, pred_offset_3D, pred_depths, batch_idxs, estimated_depth_error)

        if self.pred_features:
            assert features is not None 
            pred_features_pois = select_point_of_interest(batch, indexs, features).view(-1, features.shape[1])

        if self.consistency_eval:
           pred_corners_proj, pred_corners_proj_mask = self.anno_encoder.decode_corners_proj(pred_rotys, pred_dimensions, pred_locations, img_size_per_obj, projection_matrix_per_obj)
           consistency_score = torch.exp(-self.eval_consistency_error(pred_corners_proj, pred_corners_proj_mask, pred_keypoint_offset, img_size_per_obj,  pred_bbox_points, pred_box2d))
        #    scores.mul_(consistency_score.view(-1, 1))

        # kitti output format
        result = torch.cat([
            clses, pred_alphas, pred_box2d, 
            # change dimension back to h,w,l
            pred_dimensions.roll(shifts=-1, dims=1),
            pred_locations, pred_rotys, scores], dim=1)
        
        eval_utils = {'uncertainty_conf': uncertainty_conf,
                    'estimated_depth_error': estimated_depth_error, 'vis_scores': vis_scores,
                    'batch_indices': batch_idxs,
                    'reg_indices': indexs
        }

        if self.pred_distribution: 
            eval_utils['precision'] = precision

        if self.pred_features:
            eval_utils['features'] = pred_features_pois

        if self.consistency_eval:
            eval_utils['consistency'] = consistency_score

        # if self.ensemble:
        #     eval_utils.update({
        #         'combined_uncertainty': pred_combined_uncertainty.log(), 
        #         'combined_depth': pred_combined_depths,
        #         'discrete_position': pred_bbox_points,
        #         'xs': xs, 
        #         'ys': ys, 
        #         'batch_idxs': batch_idxs,
        #     })
        if dis_ious is not None: 
            eval_utils['dis_ious'] = dis_ious
        if depth_errors is not None: 
            eval_utils['depth_errors'] = depth_errors
        eval_utils['scores_cls'] = scores_cls
        return result, eval_utils, visualize_preds

    def get_oracle_depths(self, pred_bboxes, pred_clses, pred_combined_depths, pred_combined_uncertainty, target):
        calib = target['calib']
        pad_size = target['pad_size']
        pad_w, pad_h = pad_size

        valid_mask = target['reg_mask'].bool()
        num_gt = valid_mask.sum()
        gt_clses = target['cls_ids'][valid_mask]
        gt_boxes = target['gt_bboxes'][valid_mask]
        gt_locs = target['locations'][valid_mask]

        gt_depths = gt_locs[:, -1]
        gt_boxes_center = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2

        iou_thresh = 0.5

        # initialize with the average values
        oracle_depth = pred_combined_depths.mean(dim=1)
        estimated_depth_error = pred_combined_uncertainty.mean(dim=1)

        for i in range(pred_bboxes.shape[0]):
            # find the corresponding object bounding boxes
            box2d = pred_bboxes[i]
            box2d_center = (box2d[:2] + box2d[2:]) / 2
            img_dis = torch.sum((box2d_center.reshape(1, 2) - gt_boxes_center) ** 2, dim=1)
            same_cls_mask = gt_clses == pred_clses[i]
            img_dis[~same_cls_mask] = 9999
            near_idx = torch.argmin(img_dis)
            # iou 2d
            iou_2d = box_iou(box2d.detach().cpu().numpy(), gt_boxes[near_idx].detach().cpu().numpy())
            
            if iou_2d < iou_thresh:
                # match failed, simply choose the default average
                continue
            else:
                estimator_index = torch.argmin(torch.abs(pred_combined_depths[i] - gt_depths[near_idx]))
                oracle_depth[i] = pred_combined_depths[i,estimator_index]
                estimated_depth_error[i] = pred_combined_uncertainty[i, estimator_index]

        return oracle_depth, estimated_depth_error

    def evaluate_3D_depths(self, targets, pred_regression):
        raise NotImplementedError()

    def eval_consistency_error(self, pred_corners_proj, pred_corners_proj_mask,  pred_keypoint_offset, img_size_per_obj, pred_bbox_points, pred_box2d, eps=1e-8, alpha=1000):
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
        box_area = (pred_box2d[:, 2] - pred_box2d[:, 0]) * (pred_box2d[:, 3] - pred_box2d[:, 1])
        box_height = pred_box2d[:, 3] - pred_box2d[:, 1]
        error = error.div(box_height.clamp(eps)) * alpha
        return error 

    

def box_iou(box1, box2):
    intersection = max((min(box1[2], box2[2]) - max(box1[0], box2[0])), 0) * max((min(box1[3], box2[3]) - max(box1[1], box2[1])), 0)
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection

    return intersection / union

