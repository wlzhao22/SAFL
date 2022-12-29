import numpy as np 
from numpy import ndarray
import torch 
from mmdet.datasets.builder import PIPELINES
from mmdet3d.utils.misc import pad_ones 
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes, CameraInstance3DBoxes
from mmdet3d.core.points.lidar_points import LiDARPoints
from mmdet3d.core.points.cam_points import CameraPoints
from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
from mmdet3d.core.bbox.structures.coord_3d_mode import Coord3DMode
from mmdet3d.core.bbox import box_np_ops
from typing import Dict 
import numba 



@PIPELINES.register_module()
class GetEdgeIndices(object):
    def __init__(self, input_width, input_height,  down_ratio) -> None:
        super().__init__()
        self.output_width = input_width // down_ratio
        self.output_height = input_height // down_ratio
        self.max_edge_length = (self.output_width + self.output_height) * 2
        self.down_ratio = down_ratio
    
    def get_edge_utils(self, image_size, ori_img_size, down_ratio=4):
        img_h, img_w, _ = image_size
        ori_img_h, ori_img_w, _ = ori_img_size

        x_min, y_min = 0, 0
        x_max, y_max = (ori_img_w - 1) // down_ratio, (ori_img_h - 1) // down_ratio

        step = 1
        # boundary idxs
        edge_indices = []
        
        # left
        y = np.arange(y_min, y_max, step)
        x = np.ones(len(y)) * x_min
        
        edge_indices_edge = np.stack((x, y), axis=1).astype(np.long)
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min, None)
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min, None)
        edge_indices_edge = np.unique(edge_indices_edge, axis=0)
        edge_indices.append(edge_indices_edge)
        
        # bottom
        x = np.arange(x_min, x_max, step)
        y = np.ones(len(x)) * y_max

        edge_indices_edge = np.stack((x, y), axis=1).astype(np.long)
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min, None)
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min, None)
        edge_indices_edge = np.unique(edge_indices_edge, axis=0)
        edge_indices.append(edge_indices_edge)

        # right
        y = np.arange(y_max, y_min, -step)
        x = np.ones(len(y)) * x_max

        edge_indices_edge = np.stack((x, y), axis=1).astype(np.long)
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min, None)
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min, None)
        edge_indices_edge = np.flip(np.unique(edge_indices_edge, axis=0), axis=[0])
        edge_indices.append(edge_indices_edge)

        # top  
        x = np.arange(x_max, x_min - 1, -step)
        y = np.ones(len(x)) * y_min

        edge_indices_edge = np.stack((x, y), axis=1).astype(np.long)
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min, None)
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min, None)
        edge_indices_edge = np.flip(np.unique(edge_indices_edge, axis=0), axis=[0])
        edge_indices.append(edge_indices_edge)

        # concatenate
        edge_indices = np.concatenate([index.astype(np.long) for index in edge_indices], axis=0)

        return edge_indices

    def get_edge_indices(self, results):
        pad_shape = results['pad_shape']
        img_shape = results['img_shape']
        input_edge_indices = np.zeros([self.max_edge_length, 2], dtype=np.int64)

        # generate edge_indices for the edge fusion module
        edge_indices = self.get_edge_utils(
            pad_shape,  # actual image shape 
            img_shape,   # original image shape before padding
            self.down_ratio
        )
        input_edge_count = edge_indices.shape[0]
        input_edge_indices[:edge_indices.shape[0]] = edge_indices
        input_edge_count = input_edge_count - 1 # explain ? 

        results.update(dict(
            edge_len=input_edge_count,
            edge_indices=input_edge_indices,
        ))
        return results

    def __call__(self, results): 
        return self.get_edge_indices(results)


@PIPELINES.register_module()
class LiDARPoints2CAM(object):
    def __init__(self) -> None: 
        ... 
    
    def __call__(self, results) -> Dict: 
        points:LiDARPoints = results['points']
        assert isinstance(points, LiDARPoints) 
        calib = results['calib']
        points = points.convert_to(Coord3DMode.CAM, rt_mat=calib['R0_rect'] @ calib['Tr_velo_to_cam'])
        results['points'] = points 
        return results


@PIPELINES.register_module()
class FieldSegmentationGT(object):
    def __init__(self, N=10, size_scale=1.1, max_background_points=300) -> None:
        super().__init__()
        self.N = N
        self.size_scale = size_scale
        self.max_background_points = max_background_points
    
    def __call__(self, results) -> Dict: 
        points:CameraPoints = results['points']
        assert isinstance(points, CameraPoints)
        points:ndarray = points.tensor.numpy()[:, :3]
        boxes_3D = results['gt_bboxes_3d'].tensor.numpy().copy()
        if self.size_scale != 1: 
            boxes_3D[:, 3:6] *= self.size_scale
        calib = results['calib']
        
        lines = np.concatenate((points, -points), axis=1)
        point_indices: ndarray = box_np_ops.points_in_rbbox(
            points,
            boxes_3D,
            z_axis=1, origin=(0.5, 1, 0.5),
        )

        ret = {
            'gt_points_sampled': [],
            'gt_points_sampled_cls': []
        }
        for i in range(len(boxes_3D)):
            points_cls = point_indices[:, i].astype(dtype=np.bool)  # indicates that whether a point belongs to the instance 

            # Filter out points that do not intersect with the target sphere. 
            box_3D = boxes_3D[i]
            cx, cy, cz = box_3D[:3]
            w, l, h = box_3D[3:6]
            cy -= h * 0.5  # move the origin to center for the KITTI dataset
            ry = box_3D[6]
            radius = np.sqrt(w**2 + l**2 + h**2) * 0.5
            distances = _compute_distance_lines_point(lines, [cx, cy, cz])
            keep = distances < radius          # n_pts, 
            lines_keep = lines[keep]           # n_pts, 6
            points_keep = points[keep] # n_pts, 3
            points_cls = points_cls[keep]      # n_pts,

            # Generate background points by sampling on the line between two intersection
            # points
            background_lines = lines_keep[~points_cls]
            intersection_points = _get_intersection_lines_sphere(
                background_lines, [cx, cy, cz], radius
            ) # n_pts, 2, 3
            background_points = random_interpolate(intersection_points[:, 0, :], intersection_points[:, 1, :], self.N)  # n_pts, N, 3
            
            # Filter out occluded points
            z_mask = background_points[:, :, 2] <= background_lines[None, :, 2]
            background_points = background_points[z_mask]  # n_back, 3

            nan_mask = np.any(np.isnan(background_points), axis=1)
            background_points = background_points[~nan_mask]

            if len(background_points) > self.max_background_points:
                background_points = background_points[np.random.randint(0, len(background_points), self.max_background_points)]

            # Collect foreground points and background points. 
            points_sampled = np.concatenate((background_points, points_keep[points_cls]), axis=0)
            cls_points_sampled = np.concatenate(
                (
                    np.zeros((len(background_points),)), 
                    np.ones((len(points_sampled) - len(background_points),))
                ), 
                axis=0
            ).astype(np.bool)

            # They are converted to the object coordinate system. 
            # x, z are rotated; y is not changed.
            rot_matrix = np.array([
                [np.cos(ry), 0, -np.sin(ry)],
                [0, 1, 0],
                [np.sin(ry), 0, np.cos(ry)]
            ])
            points_sampled = (points_sampled - [cx, cy, cz]) @ rot_matrix.T
            ret['gt_points_sampled'].append(points_sampled)
            ret['gt_points_sampled_cls'].append(cls_points_sampled)
        
        results.update(ret)
        return results
            

def random_interpolate(array_a, array_b, n) -> ndarray:
    assert array_a.shape == array_b.shape 
    rand = np.random.random((n, *array_a.shape[:-1]))[..., None]
    return array_a * rand + array_b * (1 - rand)


def _compute_distance_lines_point(lines, point, eps=1e-6) -> bool: 
    lines_xyz = lines[:, :3]
    lines_direction = lines[:, 3:6]
    v = point - lines_xyz
    cross_product = np.cross(lines_direction, v)  # n, 3
    d = np.linalg.norm(cross_product, 2, axis=1) / (np.linalg.norm(lines_direction, 2, axis=1) + eps)  # n, 
    return d 


def _get_intersection_lines_sphere(lines, center, radius):
  lines = np.asarray(lines)
  center = np.asarray(center)
  lines_xyz = lines[:, :3]
  lines_direction = lines[:, 3:6]
  a = np.sum(lines_direction**2, axis=-1)                          # n, 
  b = np.sum(2 * (lines_xyz - center) * lines_direction, axis=-1)  # n, 
  c = np.sum((lines_xyz - center)**2, axis=-1) - radius**2         # n,
  delta = b**2 - 4 * a * c
  invalid_mask = delta < 0 
  delta = np.maximum(delta, 0)
  t1 = (-b + np.sqrt(delta)) / (2 * a)
  t2 = (-b - np.sqrt(delta)) / (2 * a)
  solution_1 = lines_xyz + t1[..., None] * lines_direction
  solution_2 = lines_xyz + t2[..., None] * lines_direction
  solution_1[invalid_mask] = float('nan')
  solution_2[invalid_mask] = float('nan')
  return np.stack((solution_1, solution_2), 1)