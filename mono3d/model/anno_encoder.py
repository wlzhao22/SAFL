import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.utils.misc import pad_ones

PI = np.pi


def project_image_to_rect(uv_depth, calib, camera_id: int):
	""" Input: nx3 first two channels are uv, 3rd channel
				is depth in rect camera coord.
		Output: nx3 points in rect camera coord.
	"""       
	P = calib[f'P{camera_id}']
	c_u = P[0, 2]
	c_v = P[1, 2]
	f_u = P[0, 0]
	f_v = P[1, 1]
	b_x = P[0, 3] / (-f_u)  # relative
	b_y = P[1, 3] / (-f_v)
	n = uv_depth.shape[0]
	x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
	y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y

	if isinstance(uv_depth, np.ndarray):
		pts_3d_rect = np.zeros((n, 3))
	else:
		# torch.Tensor or torch.cuda.Tensor
		pts_3d_rect = uv_depth.new(uv_depth.shape).zero_()

	pts_3d_rect[:, 0] = x
	pts_3d_rect[:, 1] = y
	pts_3d_rect[:, 2] = uv_depth[:, 2]

	return pts_3d_rect


def get_covariance_from_uncertainty(uv_depth, calib, uncertainty, camera_id: int):
	""" Input: nx3 first two channels are uv, 3rd channel
				is depth in rect camera coord.
		Output: nx3 points in rect camera coord.
	"""       
	P = calib[f'P{camera_id}']
	c_u = P[0, 2]
	c_v = P[1, 2]
	f_u = P[0, 0]
	f_v = P[1, 1]
	b_x = P[0, 3] / (-f_u)  # relative
	b_y = P[1, 3] / (-f_v)
	n = uv_depth.shape[0]
	A = uv_depth.new_empty(n, 3, 2)
	A[:, 0, 0] = (uv_depth[:, 0] - c_u) / f_u
	A[:, 0, 1] = b_x
	A[:, 1, 0] = (uv_depth[:, 1] - c_v) / f_v 
	A[:, 1, 1] = b_y 
	A[:, 2, 0] = 1. 
	A[:, 2, 1] = 0.
	x = uncertainty.new_zeros(n, 2, 2)
	x[:, 0, 0] = uncertainty

	cov = torch.bmm(A, torch.bmm(x, A.permute(0, 2, 1)))

	return cov


def get_covariance(uv_depth, calib, camera_id: int, lambda_=40):
	""" Input: nx3 first two channels are uv, 3rd channel
				is depth in rect camera coord.
		Output: nx3 points in rect camera coord.
	"""       
	P = calib[f'P{camera_id}']
	c_u = P[0, 2]
	c_v = P[1, 2]
	f_u = P[0, 0]
	f_v = P[1, 1]
	b_x = P[0, 3] / (-f_u)  # relative
	b_y = P[1, 3] / (-f_v)
	n = uv_depth.shape[0]
	A = uv_depth.new_empty(n, 3, 2)
	A[:, 0, 0] = (uv_depth[:, 0] - c_u) / f_u
	A[:, 0, 1] = b_x
	A[:, 1, 0] = (uv_depth[:, 1] - c_v) / f_v 
	A[:, 1, 1] = b_y 
	A[:, 2, 0] = 1. 
	A[:, 2, 1] = 0.
	depth = uv_depth[:, -1]
	x = uv_depth.new_zeros(n, 2, 2)
	x[:, 0, 0] = torch.exp(depth / lambda_) - 1 + 1e-1

	cov = torch.bmm(A, torch.bmm(x, A.permute(0, 2, 1)))

	var_xy = cov.new_zeros(n, 3, 3)
	var_xy[:, 0, 0] = torch.exp(depth / (lambda_ * 5)) - 1 + 1e-2
	var_xy[:, 1, 1] = torch.exp(depth / (lambda_ * 5)) - 1 + 1e-2

	return cov + var_xy

	
class Anno_Encoder():
		def __init__(self, 
        num_classes, 
        input_width, 
        input_height, 
		center_mode,
		depth_mode,
        down_ratio, 
		orientation,
		orientation_bin_size,
		heatmap_center,
		depth_range,
		depth_ref,
		dim_mean,
		dim_std,
		offset_mean, 
		offset_std,
		dim_reg,
		**kwargs
	):
			self.INF = 100000000
			self.EPS = 1e-3

			# center related
			self.num_cls = num_classes
			self.target_center_mode = heatmap_center
			# if mode == 'max', centerness is the larger value, if mode == 'area', assigned to the smaller bbox
			self.center_mode = center_mode
			
			# depth related
			self.depth_mode = depth_mode
			self.depth_range = depth_range
			self.depth_ref = torch.as_tensor(depth_ref)

			# dimension related
			self.dim_mean = torch.as_tensor(dim_mean)
			self.dim_std = torch.as_tensor(dim_std)
			self.dim_modes = dim_reg

			# orientation related
			self.alpha_centers = torch.tensor([0, PI / 2, PI, - PI / 2])
			self.multibin = (orientation == 'multi-bin')
			self.orien_bin_size = orientation_bin_size

			# offset related
			self.offset_mean = offset_mean
			self.offset_std = offset_std

			# output info
			self.down_ratio = down_ratio
			self.output_height = input_height // down_ratio
			self.output_width = input_width // down_ratio
			self.K = self.output_width * self.output_height

		@staticmethod
		def rad_to_matrix(rotys, N):
			device = rotys.device

			cos, sin = rotys.cos(), rotys.sin()

			i_temp = torch.tensor([[1, 0, 1],
								 [0, 1, 0],
								 [-1, 0, 1]]).to(dtype=torch.float32, device=device)

			ry = i_temp.repeat(N, 1).view(N, 3, 3)

			ry[:, 0, 0] *= cos
			ry[:, 0, 2] *= sin
			ry[:, 2, 0] *= sin
			ry[:, 2, 2] *= cos

			return ry

		def decode_box2d_fcos(self, centers, pred_offset, out_size=None):
			box2d_center = centers.view(-1, 2)
			box2d = box2d_center.new(box2d_center.shape[0], 4).zero_()
			# left, top, right, bottom
			box2d[:, :2] = box2d_center - pred_offset[:, :2]
			box2d[:, 2:] = box2d_center + pred_offset[:, 2:]

			N = box2d.shape[0]
			# upscale and subtract the padding
			box2d = box2d * self.down_ratio
			# clamp to the image bound
			box2d[:, 0::2] = torch.min(box2d[:, 0::2], out_size[:, [0]].to(box2d.device).float())
			box2d[:, 1::2] = torch.min(box2d[:, 1::2], out_size[:, [1]].to(box2d.device).float())
			box2d.clamp_(0)

			return box2d

		def encode_box3d(self, rotys, dims, locs):
			'''
			construct 3d bounding box for each object.
			Args:
					rotys: rotation in shape N
					dims: dimensions of objects
					locs: locations of objects

			Returns:

			'''
			if len(rotys.shape) == 2:
					rotys = rotys.flatten()
			if len(dims.shape) == 3:
					dims = dims.view(-1, 3)
			if len(locs.shape) == 3:
					locs = locs.view(-1, 3)

			device = rotys.device
			N = rotys.shape[0]
			ry = self.rad_to_matrix(rotys, N)

			# l, h, w
			dims_corners = dims.view(-1, 1).repeat(1, 8)
			dims_corners = dims_corners * 0.5
			dims_corners[:, 4:] = -dims_corners[:, 4:]
			index = torch.tensor([[4, 5, 0, 1, 6, 7, 2, 3],
								[0, 1, 2, 3, 4, 5, 6, 7],
								[4, 0, 1, 5, 6, 2, 3, 7]]).repeat(N, 1).to(device=device)
			
			box_3d_object = torch.gather(dims_corners, 1, index)
			box_3d = torch.matmul(ry, box_3d_object.view(N, 3, 8))
			box_3d += locs.unsqueeze(-1).repeat(1, 1, 8)

			return box_3d.permute(0, 2, 1)

		def decode_depth(self, depths_offset):
			'''
			Transform depth offset to depth
			'''
			if self.depth_mode == 'exp':
				depth = depths_offset.exp()
			elif self.depth_mode == 'linear':
				depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]
			elif self.depth_mode == 'inv_sigmoid':
				depth = 1 / torch.sigmoid(depths_offset) - 1
			else:
				raise ValueError

			if self.depth_range is not None:
				depth = torch.clamp(depth, min=self.depth_range[0], max=self.depth_range[1])

			return depth

		def decode_location_flatten(self, img_metas, points, offsets, depths, batch_idxs, return_centers=False):
			batch_size = len(img_metas)
			gts = torch.unique(batch_idxs, sorted=True).tolist()
			locations = points.new_zeros(points.shape[0], 3).float()
			points = (points + offsets) * self.down_ratio 

			for idx, gt in enumerate(gts):
				assert 0 <= gt < batch_size, (gt, batch_size)
				corr_pts_idx = torch.nonzero(batch_idxs == gt).squeeze(-1)
				calib = img_metas[gt]['calib']
				camera_id = img_metas[gt].get('camera_id', 2)
				# concatenate uv with depth
				corr_pts_depth = torch.cat((points[corr_pts_idx], depths[corr_pts_idx, None]), dim=1)
				locations[corr_pts_idx] = project_image_to_rect(corr_pts_depth, calib, camera_id)
			if not return_centers:
				return locations
			return locations, points


		def decode_precision_matrix(self, img_metas, points, offsets, depths, batch_idxs, depth_uncertainty, return_centers=False):
			batch_size = len(img_metas)
			gts = torch.unique(batch_idxs, sorted=True).tolist()
			precisions = points.new_zeros(points.shape[0], 3, 3).float()
			points = (points + offsets) * self.down_ratio 

			for idx, gt in enumerate(gts):
				assert 0 <= gt < batch_size, (gt, batch_size)
				corr_pts_idx = torch.nonzero(batch_idxs == gt).squeeze(-1)
				calib = img_metas[gt]['calib']
				camera_id = img_metas[gt].get('camera_id', 2)
				# concatenate uv with depth
				corr_pts_depth = torch.cat((points[corr_pts_idx], depths[corr_pts_idx, None]), dim=1)
				# cov = get_covariance_from_uncertainty(corr_pts_depth, calib, depth_uncertainty[corr_pts_idx], camera_id)
				cov = get_covariance(corr_pts_depth, calib, camera_id)
				precisions[corr_pts_idx] = torch.inverse(cov)
			if not return_centers:
				return precisions
			return precisions, points

		def decode_depth_from_keypoints_batch(self, img_metas, pred_keypoints, pred_dimensions, batch_idxs):
			# pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
			pred_height_3D = pred_dimensions[:, 1].clone()
			batch_size = len(img_metas)

			center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
			corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
			corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]

			pred_keypoint_depths = {'center': [], 'corner_02': [], 'corner_13': []}

			for idx, gt_idx in enumerate(torch.unique(batch_idxs, sorted=True).tolist()):			
				calib = img_metas[gt_idx]['calib']
				camera_id = img_metas[gt_idx].get('camera_id', 2)
				P = calib[f'P{camera_id}']
				f_u = P[0, 0]
				b_z = P[2, 3]
				corr_pts_idx = torch.nonzero(batch_idxs == gt_idx).squeeze(-1)
				center_depth = f_u * pred_height_3D[corr_pts_idx] / (F.relu(center_height[corr_pts_idx]) * self.down_ratio + self.EPS)
				corner_02_depth = f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_02_height[corr_pts_idx]) * self.down_ratio + self.EPS)
				corner_13_depth = f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_13_height[corr_pts_idx]) * self.down_ratio + self.EPS)

				corner_02_depth = corner_02_depth.mean(dim=1)
				corner_13_depth = corner_13_depth.mean(dim=1)

				pred_keypoint_depths['center'].append(center_depth - b_z)
				pred_keypoint_depths['corner_02'].append(corner_02_depth - b_z)
				pred_keypoint_depths['corner_13'].append(corner_13_depth - b_z)

			for key, depths in pred_keypoint_depths.items():
				pred_keypoint_depths[key] = torch.clamp(torch.cat(depths), min=self.depth_range[0], max=self.depth_range[1])\
					if len(depths) > 0 else pred_keypoints.new_empty((0,))

			pred_depths = torch.stack([depth for depth in pred_keypoint_depths.values()], dim=1) 

			return pred_depths

		def decode_dimension(self, cls_id, dims_offset):
			'''
			retrieve object dimensions
			Args:
					cls_id: each object id
					dims_offset: dimension offsets, shape = (N, 3)

			Returns:

			'''
			cls_id = cls_id.flatten().long()
			device = cls_id.device
			cls_dimension_mean = self.dim_mean[cls_id, :].to(device)

			if self.dim_modes[0] == 'exp':
				dims_offset = dims_offset.exp()

			if self.dim_modes[2]:
				cls_dimension_std = self.dim_std[cls_id, :].to(device)
				dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
			else:
				dimensions = dims_offset * cls_dimension_mean
				
			return dimensions

		def decode_axes_orientation(self, vector_ori, locations):
			'''
			retrieve object orientation
			Args:
					vector_ori: local orientation in [axis_cls, head_cls, sin, cos] format
					locations: object location

			Returns: for training we only need roty
							 for testing we need both alpha and roty

			'''
			if vector_ori.shape[0] == 0: 
				return vector_ori.new_empty((0,)), vector_ori.new_empty((0,))
			if self.multibin:
				pred_bin_cls = vector_ori[:, : self.orien_bin_size * 2].view(-1, self.orien_bin_size, 2)
				pred_bin_cls = torch.softmax(pred_bin_cls, dim=2)[..., 1]
				orientations = vector_ori.new_zeros(vector_ori.shape[0])
				for i in range(self.orien_bin_size):
					mask_i = (pred_bin_cls.argmax(dim=1) == i)
					s = self.orien_bin_size * 2 + i * 2
					e = s + 2
					pred_bin_offset = vector_ori[mask_i, s : e]
					orientations[mask_i] = torch.atan2(pred_bin_offset[:, 0], pred_bin_offset[:, 1]) + self.alpha_centers[i]
			else:
				axis_cls = torch.softmax(vector_ori[:, :2], dim=1)
				axis_cls = axis_cls[:, 0] < axis_cls[:, 1]
				head_cls = torch.softmax(vector_ori[:, 2:4], dim=1)
				head_cls = head_cls[:, 0] < head_cls[:, 1]
				# cls axis
				orientations = self.alpha_centers[axis_cls + head_cls * 2]
				sin_cos_offset = F.normalize(vector_ori[:, 4:])
				orientations += torch.atan(sin_cos_offset[:, 0] / sin_cos_offset[:, 1])

			locations = locations.view(-1, 3)
			rays = torch.atan2(locations[:, 0], locations[:, 2])
			alphas = orientations
			rotys = alphas + rays

			larger_idx = (rotys > PI).nonzero()
			small_idx = (rotys < -PI).nonzero()
			if len(larger_idx) != 0:
					rotys[larger_idx] -= 2 * PI
			if len(small_idx) != 0:
					rotys[small_idx] += 2 * PI

			larger_idx = (alphas > PI).nonzero()
			small_idx = (alphas < -PI).nonzero()
			if len(larger_idx) != 0:
					alphas[larger_idx] -= 2 * PI
			if len(small_idx) != 0:
					alphas[small_idx] += 2 * PI

			return rotys, alphas

		def decode_corners_proj(self, rotys, dims, locs, img_size_per_obj, projection_matrix, ):
			'''
			construct 3d bounding box for each object.
			Args:
					rotys: rotation in shape N
					dims: dimensions of objects
					locs: locations of objects

			Returns:

			'''
			if len(rotys.shape) == 2:
					rotys = rotys.flatten()
			if len(dims.shape) == 3:
					dims = dims.view(-1, 3)
			if len(locs.shape) == 3:
					locs = locs.view(-1, 3)

			device = rotys.device
			N = rotys.shape[0]
			ry = self.rad_to_matrix(rotys, N)

			# l, h, w
			dims_corners = dims.view(-1, 1).repeat(1, 2)
			dims_corners = dims_corners * 0.5
			dims_corners[:, 1:] = -dims_corners[:, 1:]
			index = torch.tensor([
				[0, 0, 1, 1, 0, 0, 1, 1,],
				[0, 0, 0, 0, 1, 1, 1, 1,],
				[0, 1, 1, 0, 0, 1, 1, 0,]
			]).repeat(N, 1).to(device)

			box_3d_object = torch.gather(dims_corners, 1, index)
			box_3d = torch.matmul(ry, box_3d_object.view(N, 3, 8))
			top_bottom = torch.zeros(N, 3, 2).to(device)
			top_bottom[:, 1, 0] = box_3d[:, 1, 0]
			top_bottom[:, 1, 1] = box_3d[:, 1, 4]
			box_3d = torch.cat((box_3d, top_bottom), axis=-1)  # n, 3, 10
			box_3d += locs.unsqueeze(-1).repeat(1, 1, 10)
			box_3d =  box_3d.permute(0, 2, 1)  # n, 10 3
			
			mask = box_3d[:, :, 2] > 0  # n, 10, 

			n = box_3d.shape[0]
			projection_matrix = projection_matrix.unsqueeze(1).expand(n, 10, 4, 4)
			box_3d = pad_ones(box_3d, -1)  # n, 10, 4
			box_3d = box_3d.unsqueeze(-2)  # n, 10, 1, 4
			pred_corners_proj = box_3d @ projection_matrix.permute(0, 1, 3, 2)
			pred_corners_proj = pred_corners_proj[..., :2] / pred_corners_proj[..., [2]]  # n, 10, 1, 2
			pred_corners_proj = pred_corners_proj.squeeze(-2) # n, 10, 2
			mask = (
				mask 
				& (pred_corners_proj[..., 0] >= 0) 
				& (pred_corners_proj[..., 1] >= 0)
				& (pred_corners_proj[..., 0] < img_size_per_obj[:, None, 0])
				& (pred_corners_proj[..., 1] < img_size_per_obj[:, None, 1])
			)
			return pred_corners_proj, mask


if __name__ == '__main__':
	pass