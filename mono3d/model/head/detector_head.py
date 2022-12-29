from typing import List
from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
import torch
from torch import nn
from torch import Tensor
import pdb
from mono3d.model.head.detector_attention_head import AttentionHead

from mono3d.model.head.get_targets import CENTER_TYPES, get_ground_truth, prepare_instances

from .detector_predictor import make_predictor
from .detector_loss import make_loss_evaluator
from .detector_infer import make_post_processor

from mmdet3d.core.bbox import LiDARInstance3DBoxes, CameraInstance3DBoxes
from mmdet3d.models import HEADS
from mmdet3d.utils.misc import pad_ones
from mmdet3d.core import box3d_multiclass_nms, xywhr2xyxyr
import numpy as np

import time 

from mono3d.model.heatmap_coder import (
	gaussian_radius,
	draw_umich_gaussian,
	draw_gaussian_1D,
	draw_ellip_gaussian,
	draw_umich_gaussian_2D,
)

PI = np.pi

@HEADS.register_module()
class MonoFlexCenterHead(nn.Module):
    def __init__(self, **kwargs):
        super(MonoFlexCenterHead, self).__init__()

        # load default arguments
        default_kwargs = {
            'num_channel': 256,
            'norm_cfg': dict(type='inplace_abn', eps=1e-3, momentum=0.1),
            'bn_momentum': 0.1,
            'init_p': 0.01,
            'edge_fusion_kernel_size': 3,
            'edge_fusion_relu': False,
            'center_sample': 'center',
            'regression_area': False,
            'corner_depth_sp': False,
            'uncertainty_range': [-10, 10],
            'loss_penalty_alpha': 2,
            'loss_beta': 4,
            'uncertainty_weight': 1.,
            'keypoint_xy_weight': [1., 1.],
            'keypoint_norm_factor': 1.,
            'depth_range': [0.1, 100],
            'depth_ref': (26.494627, 16.05988),
            'dim_mean': ((3.8840, 1.5261, 1.6286),
                               (0.8423, 1.7607, 0.6602),
                               (1.7635, 1.7372, 0.5968)),
            'dim_std': ((0.4259, 0.1367, 0.1022),
								(0.2349, 0.1133, 0.1427),
								(0.1766, 0.0948, 0.1242)),
            'offset_mean': -0.5844396972302358,
            'offset_std': 9.075032501413093,
            'cls_blacklist': None, 
            'pred_distribution': False,
            'pred_features': False,
            'consistency_eval': False,
            'attention_guided': False, 
            'seg_loss_type': 'homo',
            'mask_repair': False, 
            'seg_uncertainty_range': [0, 10],
            'quantization_method': 'floor',
        }
        for k, v in default_kwargs.items():
            if k not in kwargs:
                kwargs[k] = v
        
        self.output_width = kwargs['input_width'] // kwargs['down_ratio']
        self.output_height = kwargs['input_height'] // kwargs['down_ratio']
        self.down_ratio = kwargs['down_ratio']
        self.num_classes = kwargs['num_classes']
        self.max_objs = kwargs['max_objs']
        self.orientation_method = kwargs['orientation']
        self.multibin_size = kwargs['orientation_bin_size']
        self.consider_outside_objs = kwargs['consider_outside_objs']
        self.enable_edge_fusion = kwargs['enable_edge_fusion']
        self.proj_center_mode = kwargs['approx_3d_center']
        self.use_modify_keypoint_visible = kwargs['keypoint_visible_modify']
        self.heatmap_center = kwargs['heatmap_center']
        self.adjust_edge_heatmap = kwargs['adjust_boundary_heatmap']
        self.alpha_centers = np.array([0, PI / 2, PI, - PI / 2]) # centers for multi-bin orientation
        self.edge_heatmap_ratio = kwargs['heatmap_ratio']
        self.center_sampling = kwargs['center_sampling']
        self.attention_guided = kwargs['attention_guided']
        self.quantization_method = kwargs['quantization_method']
        
        self.predictor = make_predictor(**kwargs)
        self.loss_evaluator = make_loss_evaluator(**kwargs)
        if kwargs['attention_guided']:
            self.attention_head = AttentionHead(**kwargs)
        self.post_processor = make_post_processor(**kwargs)
        
        self.test_cfg = kwargs['test_cfg'].copy()
        self.test_cfg.relative_offset = np.asarray(self.test_cfg.relative_offset)
        self.nms_cfg = self.test_cfg['nms_cfg'].copy()

    def forward(self, inputs, outputs):
        return self.forward_single(inputs, outputs)

    def forward_test(self, inputs, outputs, depth_disturb=False, debug=False,):
        return self.forward_single(inputs, outputs, depth_disturb=depth_disturb, debug=debug)

    def forward_single(self, inputs, outputs, depth_disturb=False, debug=False):
        # t = time.time()
        x = self.predictor(inputs, outputs)
        # print('head pred: ', time.time() - t)
        # t = time.time() 
        result, aux_info, visualize_preds = self.post_processor(x, self.predictor, test=True, features=outputs['feature'], img_metas=inputs['img_metas'], depth_disturb=depth_disturb)
        if debug and hasattr(self, 'attention_head'):
            ret = self.attention_head.forward_test(self.predictor, x, inputs, dict(**outputs, reg_indices=aux_info['reg_indices']))
            aux_info.update(ret)
        # print('head post process.', time.time() - t)
        return result, aux_info, visualize_preds

    def forward_train(self, inputs, outputs, labeled=True, **kwargs):
        camera_id = inputs.get('camera_id', 2)
        prepare_instances(inputs, outputs, self.use_modify_keypoint_visible, camera_id=camera_id)
        if labeled:
            get_ground_truth(
                inputs, self.down_ratio, inputs['gt_instances'], self.num_classes, self.output_height, self.output_width, 
                self.center_sampling, 
                self.quantization_method,
                multi_center=False,
            )
        x = self.predictor(inputs, outputs)
        loss_dict, log_loss_dict = (self.loss_evaluator if labeled else self.loss_evaluator_unlabeled)(self.predictor, x, inputs, outputs)

        if self.attention_guided:
            loss_dict2 = self.attention_head(self.predictor, x, inputs, outputs)
            loss_dict.update(loss_dict2)
            log_loss_dict.update(loss_dict2)
        return loss_dict, log_loss_dict

    def encode_alpha_multibin(self, alpha, num_bin=2, margin=1 / 6):
        # encode alpha (-PI ~ PI) to 2 classes and 1 regression
        encode_alpha = np.zeros(num_bin * 2)
        bin_size = 2 * np.pi / num_bin # pi
        margin_size = bin_size * margin # pi / 6

        bin_centers = self.alpha_centers
        range_size = bin_size / 2 + margin_size

        offsets = alpha - bin_centers
        offsets[offsets > np.pi] = offsets[offsets > np.pi] - 2 * np.pi
        offsets[offsets < -np.pi] = offsets[offsets < -np.pi] + 2 * np.pi

        for i in range(num_bin):
            offset = offsets[i]
            if abs(offset) < range_size:
                encode_alpha[i] = 1
                encode_alpha[i + num_bin] = offset

        return encode_alpha

    def to_kitti_format(self, ret, img_metas, device, relative_offset=[0, 0.5, 0.], out_coord_system='lidar', **kwargs):
        batch_size = len(img_metas)
        result, aux_info, visualize_preds = ret 
        batch_idxs = aux_info['batch_indices']
        result_list: List[dict] = []
        for b in range(batch_size):
            result_b = result[batch_idxs == b]
            aux_info_b = {k: v[batch_idxs == b] for k, v in aux_info.items() if k not in ('reg_indices')}
            calib = img_metas[b]['calib']
            clses = result_b[:, 0]
            pred_alphas = result_b[:, 1]
            pred_box2d = result_b[:, 2:6]
            pred_dimensions = result_b[:, [8, 6, 7]]
            pred_locations = result_b[:, 9:12]
            pred_rotys = result_b[:, 12]
            scores = result_b[:, 13]

            pred_locations += pred_dimensions * pred_dimensions.new_tensor(relative_offset)
            
            result_dict = {}
            result_dict['boxes_3d'] = CameraInstance3DBoxes(torch.cat((pred_locations, pred_dimensions, pred_rotys.unsqueeze(-1)), axis=-1)).to(device)
            if out_coord_system == 'lidar':
                result_dict['boxes_3d'] = result_dict['boxes_3d'].convert_to(Box3DMode.LIDAR, np.linalg.inv(calib['R0_rect'] @ calib['Tr_velo_to_cam']))
            else: 
                assert out_coord_system == 'cam', out_coord_system
            result_dict['boxes'] = pred_box2d.to(device)
            result_dict['scores_3d'] = scores.to(device)
            result_dict['labels_3d'] = clses.to(device)
            result_dict['aux_info'] = aux_info_b 
            if kwargs.get('debug', False):
                result_dict['visualize_preds'] = visualize_preds
            result_list.append(result_dict)
        return result_list

    def nms_3d(self, results, device, out_device='cpu'): 
        boxes_3d = results['boxes_3d']
        n = boxes_3d.tensor.shape[0] 
        boxes_3d_for_nms = xywhr2xyxyr(boxes_3d.bev)
        scores = results['scores_3d']
        categories = results['labels_3d']
        indices = torch.tensor(list(range(n))).long()
        # scores = torch.scatter(
        #     torch.zeros(scores.shape[0], int(categories.max().item()) + 2).to(device), 
        #     dim=1, 
        #     index=categories.long().reshape((n, 1)),
        #     src=scores.reshape((n, 1))
        # )
        # location_score = results['location_score']
        score_for_nms = scores
        score_for_nms = torch.stack((score_for_nms, -score_for_nms), -1)
        if n != 0:

            bboxes, _, _, selected, *_ = box3d_multiclass_nms(
                boxes_3d.tensor.to(device), boxes_3d_for_nms.to(device), 
                score_for_nms.to(device), self.test_cfg['nms_cfg']['score_threshold'], self.test_cfg['max_per_img'],
                self.nms_cfg,
                mlvl_attr_scores=indices.to(device) # hack: get the selected indices 
            )
            selected = selected.long()
        else: 
            selected = []
            bboxes = boxes_3d

        results['boxes_3d'] = type(boxes_3d)(bboxes).to(out_device)
        results['scores_3d'] = scores[selected].to(out_device)
        results['labels_3d'] = categories[selected].to(out_device) 
        for k in results.keys():
            if k in ('boxes_3d', 'scores_3d', 'labels_3d', 'eval_utils', 'aux_info', 'visualize_preds'):
                continue 
            results[k] = results[k][selected].to(out_device)
        if 'aux_info' in results:
            aux_info = results['aux_info']
            for k in aux_info.keys():
                aux_info[k] = aux_info[k][selected].to(out_device)
        if 'visualize_preds' in results:
            visualize_preds = results['visualize_preds']
            for k in visualize_preds.keys():
                if k in ('heat_map', 'depth_uncertainty'): continue
                # print(k, visualize_preds[k].shape, selected)
                visualize_preds[k] = visualize_preds[k][selected].to(out_device)
        return results  

    def result_sampling(self, results, img_metas, sample_depth=[-2, -1, -0.5, 0, 0.5, 1, 2], lambda_=80, depth_thresh=20, score_rescale=True,):
        calib = img_metas[0]['calib']
        P = calib['P2']
        f_u, f_v = P[0, 0], P[1, 1]
        c_u, c_v = P[0, 2], P[1, 2]
        r_u, r_v = P[0, 3], P[1, 3]
        def sample(x, y, z, dz):
            du = (f_u * x - r_u) / z 
            dv = (f_v * y - r_v) / z
            dx = du * dz / f_u 
            dy = dv * dz / f_v 
            return dx + x, dy + y, z + dz 
        
        boxes_3d = []
        boxes = []
        scores_3d = []
        labels_3d = []

        boxes_source = results['boxes_3d'].convert_to(Box3DMode.CAM, calib['R0_rect'] @ calib['Tr_velo_to_cam'])
        scores_cls = results['aux_info']['scores_cls']
        for i in range(len(results['boxes_3d'])):
            x, y, z = boxes_source.tensor[i, :3]
            if z > self.test_cfg.max_depth:
                continue 
            if z < depth_thresh: 
                list_dz = [0]
            else:
                list_dz = sample_depth 
            for dz in list_dz:
                x2, y2, z2 = sample(x, y, z, dz)
                boxes_3d.append([x2, y2, z2, *boxes_source.tensor[i, 3:]])
                boxes.append(results['boxes'][i].numpy())
                if score_rescale:
                    sigma = torch.exp(z2 / lambda_)
                    scale = torch.exp(-dz**2 / sigma**2)
                    scores_3d.append(results['scores_3d'][i].item() * scale)
                    # scores_3d.append(scores_cls[i] * scale)
                else:
                    scores_3d.append(results['scores_3d'][i].item())
                labels_3d.append(results['labels_3d'][i].item())
        return {
            'boxes_3d': CameraInstance3DBoxes(torch.tensor(np.asarray(boxes_3d)))\
            .convert_to(Box3DMode.LIDAR, np.linalg.inv(calib['R0_rect'] @ calib['Tr_velo_to_cam'])),
            'boxes': torch.tensor(np.asarray(boxes)),
            'scores_3d': torch.tensor(np.asarray(scores_3d)),
            'labels_3d': torch.tensor(np.asarray(labels_3d)),
        }


def approx_proj_center(proj_center, surface_centers, img_size):
    # surface_inside
    img_w, img_h = img_size
    surface_center_inside_img = (surface_centers[:, 0] >= 0) & (surface_centers[:, 1] >= 0) & \
                            (surface_centers[:, 0] <= img_w - 1) & (surface_centers[:, 1] <= img_h - 1)

    if surface_center_inside_img.sum() > 0:
        target_surface_center = surface_centers[surface_center_inside_img.argmax()]
        # y = ax + b
        a, b = np.polyfit([proj_center[0], target_surface_center[0]], [proj_center[1], target_surface_center[1]], 1)
        valid_intersects = []
        valid_edge = []

        left_y = b
        if (0 <= left_y <= img_h - 1):
            valid_intersects.append(np.array([0, left_y]))
            valid_edge.append(0)

        right_y = (img_w - 1) * a + b
        if (0 <= right_y <= img_h - 1):
            valid_intersects.append(np.array([img_w - 1, right_y]))
            valid_edge.append(1)

        top_x = -b / a
        if (0 <= top_x <= img_w - 1):
            valid_intersects.append(np.array([top_x, 0]))
            valid_edge.append(2)

        bottom_x = (img_h - 1 - b) / a
        if (0 <= bottom_x <= img_w - 1):
            valid_intersects.append(np.array([bottom_x, img_h - 1]))
            valid_edge.append(3)

        valid_intersects = np.stack(valid_intersects)
        min_idx = np.argmin(np.linalg.norm(valid_intersects - proj_center.reshape(1, 2), axis=1))
        
        return valid_intersects[min_idx], valid_edge[min_idx]
    else:
        return None


def bulid_head(cfg, in_channels):
    
    return MonoFlexCenterHead(cfg, in_channels)