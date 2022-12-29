
import numpy as np
from numpy import ndarray
import torch 
from mmdet3d.core.bbox import LiDARInstance3DBoxes, CameraInstance3DBoxes
from mmdet3d.utils.misc import pad_ones
from mmdet3d.models.utils.utils_2d.instances import Instances
from mmdet3d.models.utils.utils_2d.boxes import Boxes
from typing import List, Tuple

from mono3d.model.heatmap_coder import draw_umich_gaussian, gaussian_radius
from mono3d.model.heatmap_coder import draw_umich_gaussian_2D


INF = 100000000 
CENTER_TYPES = ['center', 'top', 'bottom', 'edge1', 'edge2', 'edge3', 'edge4']
num_center_types = len(CENTER_TYPES) + 1

def _compute_grids(h, w, stride, device, quantization_method):
    grids = []
    shifts_x = torch.arange(
        0, w, 
        step=stride,
        dtype=torch.float32, device=device)
    shifts_y = torch.arange(
        0, h, 
        step=stride,
        dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    grids_per_level = torch.stack((shift_x, shift_y), dim=1)
    if quantization_method == 'floor': 
        grids_per_level += stride // 2
    grids.append(grids_per_level)
    return grids


def _get_target_center(obj_3d: ndarray, corners_3d: ndarray, P: ndarray, t: str) -> Tuple[ndarray, int]:
    assert t in ('center', 'top', 'bottom', 'edge1', 'edge2', 'edge3', 'edge4')

    if t in ('center', 'top', 'bottom'):
        loc = obj_3d[:3]
        dim = obj_3d[3:6]
        if t == 'top': 
            loc[1] = loc[1] - dim[1] / 2 
        elif t == 'bottom':
            loc[1] = loc[1] + dim[1] / 2
    elif t in ('edge1', 'edge2', 'edge3', 'edge4'):
        index = int(t[4]) - 1
        pair = corners_3d[[0 + index, 4 + index], :]  # shape=(2, 3)
        loc = pair.sum(0) / 2  # shape=(3,)
    else:
        raise RuntimeError("unsupported center type: ", t)
    
    proj_loc = P @ pad_ones(loc, dim=0) 
    proj_loc = proj_loc[:2] / proj_loc[2]
    return proj_loc, CENTER_TYPES.index(t)


def get_ground_truth(
    inputs, down_ratio, 
    gt_instances,
    num_classes,
    output_height, 
    output_width,
    center_sampling,
    quantization_method,
    multi_center: bool,
    edge_heatmap_ratio=0.5,
):
    batch_size = len(inputs['img_metas'])
    device = inputs['img'].device
    quantization_func = _quantization_round if quantization_method == 'round' else _quantization_floor

    
    img_h, img_w, *_ = inputs['img_metas'][0]['pad_shape']

    grids = _compute_grids(img_h, img_w, down_ratio, device, quantization_method=quantization_method)[0]
    M = grids.shape[0]

    heatmap = np.zeros((batch_size, num_classes, output_height, output_width))
    heatmap_center_type = np.zeros((batch_size, num_center_types, output_height, output_width))
    instance_id_start = 0
    instance_ids = []
    for b in range(batch_size):
        N = len(gt_instances[b])
        centers = gt_instances[b].gt_centers
        centers_expanded = centers.view(1, N, 2).expand(M, N, 2)
        centers_discret = (quantization_func(centers_expanded / down_ratio) * down_ratio).float()
        if quantization_method == 'floor': 
            centers_discret += down_ratio / 2 
        is_peak = (((grids.view(M, 1, 2).expand(M, N, 2) - centers_discret) ** 2).sum(dim=2) == 0)
        is_center3x3 = _get_center3x3_d(
            grids, centers_discret, down_ratio
        )
        reg_mask = is_center3x3 if center_sampling else is_peak
        
        dist2 = ((grids.view(M, 1, 2).expand(M, N, 2) - centers_expanded) ** 2).sum(dim=2)  # M x N
        dist2[is_peak] = 0
        instance_id = _get_target_ids(
            dist2, reg_mask, gt_instances[b], id_start=instance_id_start
        )
        instance_id_start += len(gt_instances[b])
        instance_ids.append(instance_id)

        for i in range(N):
            box2d = gt_instances[b].gt_boxes.tensor[i].cpu() / down_ratio
            trunc_mask = gt_instances[b].gt_trunc_mask.cpu()
            bbox_dim = box2d[2:] - box2d[:2]
            cls_id = gt_instances[b].gt_classes[i]
            center_type_id = gt_instances[b].gt_center_labels[i] if multi_center else 0
            gaussian_center = quantization_func(gt_instances[b].gt_centers[i] / down_ratio)

            if not trunc_mask[i]:
                # for inside objects, generate circular heatmap
                radius = gaussian_radius(bbox_dim[1], bbox_dim[0])
                radius = max(0, int(radius))
                _ = draw_umich_gaussian(heatmap[b, cls_id], gaussian_center, radius)
                if multi_center:
                    _ = draw_umich_gaussian(heatmap_center_type[b, center_type_id], gaussian_center, radius)
            else: 
                # for outside objects, generate 1-dimensional heatmap
                bbox_width = min(gaussian_center[0] - box2d[0], box2d[2] - gaussian_center[0])
                bbox_height = min(gaussian_center[1] - box2d[1], box2d[3] - gaussian_center[1])
                radius_x, radius_y = bbox_width * edge_heatmap_ratio, bbox_height * edge_heatmap_ratio
                radius_x, radius_y = max(0, int(radius_x)), max(0, int(radius_y))
                assert min(radius_x, radius_y) == 0
                _ = draw_umich_gaussian_2D(heatmap[b, cls_id], gaussian_center, radius_x, radius_y)
                if multi_center:
                    _ = draw_umich_gaussian_2D(heatmap_center_type[b, center_type_id], gaussian_center, radius_x, radius_y)
                
    inputs['grids'] = grids
    inputs['gt_heatmap'] = torch.tensor(
        heatmap if not multi_center else 
        np.concatenate((heatmap, heatmap_center_type), axis=1)  # multi_center
    ).to(device)
    inputs['gt_instance_ids'] = torch.cat(instance_ids)


def prepare_instances(inputs, outputs, use_modify_keypoint_visible, camera_id, center_type='center'):
    device = inputs['img'].device
    gt_instances: List[Instances] = []
    batch_size = len(inputs['img_metas'])
    for b in range(batch_size):
        h, w, _ = inputs['img_metas'][b]['img_shape']
        instances_per_image = Instances((h, w))

        calib = inputs['img_metas'][b]['calib']
        project_matrix = calib[f'P{camera_id}']

        objs_3d = inputs['gt_bboxes_3d'][b] if 'gt_bboxes_3d' in inputs else CameraInstance3DBoxes(torch.empty((0, 7)))
        if isinstance(objs_3d, list): objs_3d = objs_3d[0]
        objs_2d = inputs['gt_bboxes_2d'][b] if 'gt_bboxes_2d' in inputs else torch.empty(0, 4)
        if isinstance(objs_2d, list): objs_2d = objs_2d[0]
        img_h, img_w, _ = inputs['img_metas'][b]['img_shape']  # smaller than actual size, e.g., (375, 1242)

        n_objs = len(objs_2d)
        assert n_objs == objs_3d.tensor.shape[0]
        
        keypoints = torch.zeros(n_objs, 10, 3)
        keypoints_mask = torch.zeros(n_objs, 10).bool()
        keypoints_depth_mask = torch.zeros(n_objs, 3).bool()
        locations = torch.zeros(n_objs, 3)
        dimensions = torch.zeros(n_objs, 3)
        alphas = torch.zeros(n_objs, 1)
        orientation = torch.zeros(n_objs, 8)
        keep = torch.zeros((n_objs,), dtype=torch.bool)
        proj_centers = torch.zeros((n_objs, 2), dtype=torch.float)
        target_center_labels = torch.zeros((n_objs, ), dtype=torch.long)
        target_centers = torch.zeros((n_objs, 2), dtype=torch.float)
        trunc_mask = torch.zeros([n_objs, ], dtype=torch.bool)
        occluded = torch.zeros([n_objs, ], dtype=torch.long)
        for i in range(n_objs):
            obj_2d = objs_2d[i].cpu()  # shape: (4, )
            obj_3d = objs_3d[i].tensor.flatten().cpu()  # shape: (7, )
            cls_id = inputs['gt_labels'][b][i]
            keep[i] = cls_id >= 0 and obj_3d[2] >= 0

            # bottom centers ==> 3D centers
            locs = obj_3d[:3]
            dim = obj_3d[3:6]
            locs[1] = locs[1] - dim[1] / 2

            # generate 8 corners of 3d bbox
            corners_3d = objs_3d[i].corners.view(8, 3)[[6, 7, 3, 2, 5, 4, 0, 1]].numpy()  # shape: (8, 3)
            corners_2d = pad_ones(corners_3d, dim=1) @ project_matrix.T
            corners_2d = corners_2d[..., :2] / corners_2d[..., [2]]
            box2d = obj_2d.numpy().copy()
            box2d_center = (box2d[:2] + box2d[2:]) / 2.

            # project 3d location to the image plane
            proj_center = project_matrix @ pad_ones(locs, dim=0).numpy()
            proj_center = proj_center[:2] / proj_center[[2]]        
            
            # 10 keypoints
            bot_top_centers = np.stack((corners_3d[:4].mean(axis=0), corners_3d[4:].mean(axis=0)), axis=0)
            keypoints_3D = np.concatenate((corners_3d, bot_top_centers), axis=0)
            keypoints_2D = pad_ones(keypoints_3D, dim=-1) @ project_matrix.T
            keypoints_2D = keypoints_2D[..., :2] / keypoints_2D[..., [2]]

            target_center, center_label = _get_target_center(obj_3d.numpy(), corners_3d, project_matrix, center_type)

            # generate approximate projected center when it is outside the image
            proj_inside_img = (0 <= target_center[0] <= img_w - 1) & (0 <= target_center[1] <= img_h - 1)            
            if not proj_inside_img:
                ret = approx_proj_center(
                    target_center, box2d_center[None, :], (img_w, img_h)
                )
                if ret is None: 
                    keep[i] = False 
                    continue 
                target_center = ret[0]
            else:
                ...
            target_center[0] = np.clip(target_center[0], 0, img_w - 1)
            target_center[1] = np.clip(target_center[1], 0, img_h - 1)

            # keypoints mask: keypoint must be inside the image and in front of the camera
            keypoints_x_visible = (keypoints_2D[:, 0] >= 0) & (keypoints_2D[:, 0] <= img_w - 1)
            keypoints_y_visible = (keypoints_2D[:, 1] >= 0) & (keypoints_2D[:, 1] <= img_h - 1)
            keypoints_z_visible = (keypoints_3D[:, -1] > 0)

            # xyz visible
            keypoints_visible = keypoints_x_visible & keypoints_y_visible & keypoints_z_visible
            # center, diag-02, diag-13
            keypoints_depth_valid = np.stack((keypoints_visible[[8, 9]].all(), keypoints_visible[[0, 2, 4, 6]].all(), keypoints_visible[[1, 3, 5, 7]].all()))

            if use_modify_keypoint_visible:
                keypoints_visible = np.append(np.tile(keypoints_visible[:4] | keypoints_visible[4:8], 2), np.tile(keypoints_visible[8] | keypoints_visible[9], 2))
                keypoints_depth_valid = np.stack((keypoints_visible[[8, 9]].all(), keypoints_visible[[0, 2, 4, 6]].all(), keypoints_visible[[1, 3, 5, 7]].all()))

                keypoints_visible = keypoints_visible.astype(np.float32)
                keypoints_depth_valid = keypoints_depth_valid.astype(np.float32)
            
            bbox_dim = box2d[2:] - box2d[:2]
            rot_y = obj_3d[-1]
            alpha = -torch.atan2(obj_3d[0], obj_3d[2]) + rot_y  # shape: (n,)
            if alpha < -np.pi: alpha += np.pi * 2
            if alpha > np.pi: alpha -= np.pi * 2 
            assert -np.pi <= alpha <= np.pi 
            
            proj_centers[i, :] = torch.tensor(proj_center)
            keypoints[i, :, :2] = torch.tensor(keypoints_2D)
            keypoints[i, :, 2] = torch.tensor(keypoints_visible)
            keypoints_mask[i, :] = torch.tensor(keypoints_visible)
            keypoints_depth_mask[i] = torch.tensor(keypoints_depth_valid)
            locations[i] = locs
            dimensions[i, :] = dim
            alphas[i] = alpha
            orientation[i] = torch.tensor(encode_alpha_multibin(alpha, num_bin=4))
            target_centers[i, :] = torch.tensor(target_center)
            target_center_labels[i] = center_label
            occluded[i] = inputs['gt_occluded'][b][i] if 'gt_occluded' in inputs else 0
            trunc_mask[i] = not proj_inside_img

        instances_per_image.gt_centers = target_centers.to(device)
        instances_per_image.gt_center_labels = target_center_labels.to(device)
        instances_per_image.gt_project_center = proj_centers.to(device)
        instances_per_image.gt_boxes = Boxes(objs_2d)
        instances_per_image.gt_classes = inputs['gt_labels'][b]
        instances_per_image.gt_boxes_3d = objs_3d.to(device)
        instances_per_image.gt_depth = objs_3d.tensor[:, 2].to(device)
        instances_per_image.gt_alpha = alphas.to(device)
        instances_per_image.gt_locations = locations.to(device)
        instances_per_image.gt_orientation = orientation.to(device)
        instances_per_image.gt_keypoints = keypoints.to(device)
        instances_per_image.gt_keypoints_mask = keypoints_mask.to(device)
        instances_per_image.gt_keypoints_depth_mask = keypoints_depth_mask.to(device)
        instances_per_image.gt_trunc_mask = trunc_mask.to(device)
        instances_per_image.gt_occluded = occluded.to(device)
        instances_per_image = instances_per_image[keep]
        gt_instances.append(instances_per_image)
    if 'gt_instances' in inputs:
        assert len(gt_instances) == len(inputs['gt_instances'])  # check batch size 
        inputs['gt_instances'] = [Instances.cat([a, b]) for a, b in zip(inputs['gt_instances'], gt_instances)]
    else:
        inputs['gt_instances'] = gt_instances


def encode_alpha_multibin(alpha, num_bin=2, margin=1 / 6):
    alpha_centers = np.array([0, np.pi / 2, np.pi, - np.pi / 2]) # centers for multi-bin orientation
    # encode alpha (-PI ~ PI) to 2 classes and 1 regression
    encode_alpha = np.zeros(num_bin * 2)
    bin_size = 2 * np.pi / num_bin # pi
    margin_size = bin_size * margin # pi / 6

    bin_centers = alpha_centers
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


def _get_center3x3_d(locations, centers_discret_expand, down_ratio):
    '''
    Inputs:
        locations: M x 2
        centers: N x 2
    '''
    M = locations.shape[0]
    M, N, _ = centers_discret_expand.shape
    locations_expanded = locations.view(M, 1, 2).expand(M, N, 2) # M x N x 2
    dist_x = (locations_expanded[:, :, 0] - centers_discret_expand[:, :, 0]).abs()
    dist_y = (locations_expanded[:, :, 1] - centers_discret_expand[:, :, 1]).abs()
    return (dist_x <= down_ratio) & \
        (dist_y <= down_ratio)


def _get_target_ids(dist, mask, gt_instance, id_start):
    M, N = mask.shape 
    dist[mask == 0] = INF * 1.0 
    min_dist, min_inds = dist.min(dim=1)  # M 
    instance_ids = torch.arange(0, len(gt_instance)).long().unsqueeze(-1).to(mask.device)  # N x 1 
    instance_ids = instance_ids.expand(M, N, 1)
    instance_ids_per_im = instance_ids[range(M), min_inds] + id_start 
    instance_ids_per_im[min_dist == INF] = -1 
    return instance_ids_per_im


def _quantization_round(tensor):
    if isinstance(tensor, torch.Tensor): 
        return tensor.round().int()
    elif isinstance(tensor, np.ndarray): 
        return np.round(tensor).astype(np.int)
    else: 
        raise NotImplementedError() 


def _quantization_floor(tensor): 
    if isinstance(tensor, torch.Tensor): 
        return tensor.int() 
    elif isinstance(tensor, np.ndarray): 
        return tensor.astype(np.int)
    else: 
        raise NotImplementedError()
