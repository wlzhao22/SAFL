# modified from https://raw.githubusercontent.com/xingyizhou/CenterNet2/master/detectron2/structures/masks.py
# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools

from torch.tensor import Tensor
from mmdet3d.utils.misc import pad_ones
import numpy as np
from numpy import ndarray
from typing import Any, Dict, Iterator, List, Tuple, Union
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F

from mmdet3d.models.roi_heads.roi_align import ROIAlign
from mmdet3d.core.bbox import CameraInstance3DBoxes
from mmdet3d.core.bbox import box_np_ops as box_np_ops


class InstancePoints(object):    
    def __init__(self, points: torch.Tensor, instance_boxes: CameraInstance3DBoxes):
        """
        Args:
            point_ids: tensor of N, representing N instances in the image.
        """
        device = points.device if isinstance(points, torch.Tensor) else torch.device("cpu")
        self.points = points 
        self.instance_boxes = instance_boxes

    def to(self, *args: Any, **kwargs: Any) -> "InstancePoints":
        return InstancePoints(self.points.to(*args, **kwargs), self.instance_boxes.to(*args, **kwargs))

    @property
    def device(self) -> torch.device:
        return self.points.device

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "InstancePoints":
        """
        """
        return InstancePoints(self.points, self.instance_boxes[item])

    def __iter__(self) -> torch.Tensor:
        raise NotImplementedError()

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self))
        return s

    def __len__(self) -> int:
        return len(self.instance_boxes)

    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.

        Returns:
            Tensor: a BoolTensor which represents
                whether each mask is empty (False) or non-empty (True).
        """
        raise NotImplementedError()

    def crop_and_resize(self, boxes: Tensor, mask_size: int, vu: Tensor, calib: Dict) -> Tuple[torch.Tensor, ...]:
        '''
        inputs: 
            - boxes: shape=(n, 4)
            - mask_size: 
            - vu: shape=(n, 2, mask_size, mask_size)
        '''
        assert len(boxes) == len(self), "{} != {}".format(len(boxes), len(self))
        device = self.points.device
        n = len(boxes)
        P = calib['P2']
        f_u, f_v = P[0, 0], P[1, 1]
        c_u, c_v = P[0, 2], P[1, 2]
        r_u, r_v = P[0, 3], P[1, 3]
        uv = vu[:, [1, 0]].permute(0, 2, 3, 1)  # shape=(n, mask_size, mask_size, 2)
            
        out_xyz = torch.full((n, mask_size, mask_size, 3), fill_value=-1).to(device)
        out_box3d = torch.full((n, mask_size, mask_size, 7), fill_value=-1).to(device) 
        out_offset = torch.full((n, mask_size, mask_size, 3),fill_value=-1).to(device)
        out_cls = torch.full((n, mask_size, mask_size, 1), fill_value=-1).to(device)

        point_indices: ndarray = box_np_ops.points_in_rbbox(
            self.points.cpu().numpy(), 
            self.instance_boxes.tensor.cpu().numpy(), 
            z_axis=1, origin=(0.5, 1, 0.5)
        )
        point_indices: Tensor = self.points.new_tensor(point_indices)
        points_proj = pad_ones(self.points[:, :3], dim=1) @ self.points.new_tensor(calib['P2'].T)
        points_proj = points_proj[:, :2] / points_proj[:, [2]]

        for i in range(n):
            l, t, r, b = boxes[i]
            w, h = r - l, b - t 
            points_cls = point_indices[:, i].bool()  # indicates that whether a point belongs to the instance
            # All coordinates are normalized by the box size
            if points_cls.numel() == 0: continue 
            points_x = (points_proj[:, 0] - l) / w
            points_y = (points_proj[:, 1] - t) / h
            # Points out of the box are filtered out.
            keep = (0 <= points_x) & (points_x < 1) & (0 <= points_y) & (points_y < 1)
            points_x = points_x[keep]      # m
            points_y = points_y[keep]      # m
            points_cls = points_cls[keep]  # m
            frustum_pts_uv = points_proj[keep]  # m, 2
            points_frustum = self.points[keep]  # m, 4
            if points_cls.numel() == 0: continue 

            m = len(points_x)
            distance = (
                frustum_pts_uv.view(1, 1, m, 2) - uv[i].unsqueeze(2)
            ).pow(2).sum(-1)
            assert distance.shape == (mask_size, mask_size, m)

            distance_flatten = distance.view(mask_size * mask_size, m)
            min_ind = distance_flatten.argmin(0)   # m
            # min_distance = distance_flatten[min_ind]
            unique_min_ind, unique_selection = unique(min_ind, 0)
            # min_distance = min_distance[unique_selection] 
            points_cls = points_cls[unique_selection]
            points_frustum = points_frustum[unique_selection]
            row = unique_min_ind.div(mask_size).long() 
            col = (unique_min_ind - row * mask_size).long()
            out_cls[i, row, col, 0] = points_cls.float()
            out_xyz[i, row, col] = points_frustum[:, :3]
            
        out_xyz[..., 0] = ((uv[..., 0] - c_u) * out_xyz[..., 2] - r_u) / f_u
        out_xyz[..., 1] = ((uv[..., 1] - c_v) * out_xyz[..., 2] - r_v) / f_v
        obj_x = self.instance_boxes.tensor[:, 0].unsqueeze(-1).unsqueeze(-1).to(device)
        obj_y = (self.instance_boxes.tensor[:, 1] - self.instance_boxes.tensor[:, 4].div(2)).unsqueeze(-1).unsqueeze(-1).to(device)
        obj_z = self.instance_boxes.tensor[:, 2].unsqueeze(-1).unsqueeze(-1).to(device)
        out_offset[..., 0] = out_xyz[..., 0] - obj_x
        out_offset[..., 1] = out_xyz[..., 1] - obj_y
        out_offset[..., 2] = out_xyz[..., 2] - obj_z
        # out_offset[..., 3] = ((uv[..., 0] - c_u) * obj_z - r_u) / f_u - obj_x
        # out_offset[..., 4] = ((uv[..., 1] - c_v) * obj_z - r_v) / f_v - obj_y
        
        out = [out_xyz, out_offset, out_cls]
        out = (o.permute(0, 3, 1, 2) for o in out)
        return tuple(out)


def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(0, inverse, perm)


def mean_filter(tensor, mask, size=(3, 1), padding=(1, 0), eps=1e-8):
    '''
    tensor: shape=(bs, h, w, 1)
    mask: shape=(bs, h, w, 1)
    '''
    tensor = tensor.permute(0, 3, 1, 2).contiguous()  # shape=(bs, 1, h, w)
    mask2 = mask.permute(0, 3, 1, 2).contiguous().float()  # shape=(bs, 1, h, w)
    kernel = tensor.new_ones(1, 1, *size) 
    sum = F.conv2d(tensor * mask2, kernel, padding=padding)
    n = F.conv2d(mask2, kernel, padding=padding)
    mean = sum / n.add(eps)
    return mean.permute(0, 2, 3, 1) * mask


