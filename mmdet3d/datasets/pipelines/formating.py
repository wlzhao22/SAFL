import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor

PIPELINES._module_dict.pop('DefaultFormatBundle')


@PIPELINES.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            img = results['img']
            # add default meta keys
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)

            for k in results['img_fields']:
                if not (k.startswith('img_prev') or k == 'img_r'): continue
                img_k = results[k]
                if len(img_k.shape) < 3:
                    img_k = np.expand_dims(img_k.transpose(2, 0, 1))
                img_k = np.ascontiguousarray(img_k.transpose(2, 0, 1))
                results[k] = DC(to_tensor(img_k), stack=True)

            if 'img_pre' in results and 'img_post' in results:
                img_pre = results['img_pre']
                if len(img_pre.shape) < 3:
                    img_pre = np.expand_dims(img_pre, -1)
                img_pre = np.ascontiguousarray(img_pre.transpose(2, 0, 1))
                results['img_pre'] = DC(to_tensor(img_pre), stack=True)

                img_post = results['img_post']
                if len(img_post.shape) < 3:
                    img_post = np.expand_dims(img_post, -1)
                img_post = np.ascontiguousarray(img_post.transpose(2, 0, 1))
                results['img_post'] = DC(to_tensor(img_post), stack=True)

            if 'img_aug' in results:
                img_aug = results['img_aug']
                if len(img_aug.shape) < 3:
                    img_aug = np.expand_dims(img_aug, -1)
                img_aug = np.ascontiguousarray(img_aug.transpose(2, 0, 1))
                results['img_aug'] = DC(to_tensor(img_aug), stack=True)

            if 'img_affine' in results:
                img_affine = results['img_affine']
                if len(img_affine.shape) < 3:
                    img_affine = np.expand_dims(img_affine, -1)
                img_affine = np.ascontiguousarray(img_affine.transpose(2, 0, 1))
                results['img_affine'] = DC(to_tensor(img_affine), stack=True)

            if 'img_c' in results:
                img_c = results['img_c']
                if len(img_c.shape) < 3:
                    img_c = np.expand_dims(img_c, -1)
                img_c = np.ascontiguousarray(img_c.transpose(2, 0, 1))
                results['img_c'] = DC(to_tensor(img_c), stack=True)
        for key in results.get('flow_fields', []):
            flow = results[key]
            flow = np.ascontiguousarray(flow.transpose(2, 0, 1))
            results[key] = DC(to_tensor(flow), stack=True)
        for key in [
                'proposals', 'gt_bboxes_2d', 'gt_bboxes_ignore', 'gt_labels',
                'gt_reids', 'gt_wheels', 'gt_wheels_exist', "gt_ddd_face_clzs",
                "gt_ddd_short_ys", "gt_ddd_short_xs", "position", "gt_ddd_sizes",
                "gt_ddd_rotations", "gt_ddd_lfs", "gt_ddd_rfs", "gt_ddd_rrs", "gt_ddd_lrs",
                "corner_pts", "corner_pts_affine", "match_corner_pts", "negative_corner_pts",
                "negative_affine_corner_pts", "lane_center", "lane_marker", "lane_label",
                "gt_reids_c", "gt_bboxes_2d_c", "gt_labels_c", 'gt_ddds_head_direction',
                'gt_ddds_dx', 'gt_ddds_dw', 'gt_ddds_l0', 'gt_ddds_l1', 'gt_ddds_l2',
                'gt_ddds_res_depth', 'gt_ddds_rotation', 'gt_ddds_size', 'gt_ddds_center_2d',
                'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers2d', 'depths', 
                'gt_occluded',
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        if 'gt_bboxes_3d' in results:
            if isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = DC(
                    results['gt_bboxes_3d'], cpu_only=True)
            else:
                results['gt_bboxes_3d'] = DC(
                    to_tensor(results['gt_bboxes_3d']))

        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        if 'gt_depth' in results:
            results['gt_depth'] = DC(
                to_tensor(results['gt_depth'][None, ...]), stack=True)
        if 'gt_free_space' in results:
            results['gt_free_space'] = DC(
                to_tensor(results['gt_free_space'][None, ...]), stack=True)
        if 'gt_lane_seg' in results:
            results['gt_lane_seg'] = DC(
                to_tensor(results['gt_lane_seg'][None, ...]), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__

@PIPELINES.register_module()
class Collect2D(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:

            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'

@PIPELINES.register_module()
class Collect3D(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:

            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'rect': rectification matrix
        - 'Trv2c': transformation from velodyne to camera coordinate
        - 'P2': transformation betweeen cameras
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img', \
            'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip', \
            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', \
            'img_norm_cfg', 'rect', 'Trv2c', 'P2', 'pcd_trans', \
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'pad_shape', 'scale_factor', 'flip',
                            'cam_intrinsic', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'rect', 'Trv2c', 'P2', 'pcd_trans',
                            'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                            'pts_filename', 'transformation_3d_flow')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
        try:
            data = {}
            img_metas = {}
            for key in self.meta_keys:
                if key in results:
                    img_metas[key] = results[key]

            data['img_metas'] = DC(img_metas, cpu_only=True)
            for key in self.keys:
                data[key] = results[key]
            return data
        except KeyError as e:
            print("avaliable keys: {}".format(results.keys()))
            raise e

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'


@PIPELINES.register_module()
class DefaultFormatBundle3D(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        super(DefaultFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = DC(results['points'].tensor)

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

        if self.with_gt:
            # Clean GT bboxes in the final
            if 'gt_bboxes_3d_mask' in results:
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][
                    gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][
                        gt_bboxes_3d_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][
                        gt_bboxes_3d_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                 dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                                                    dtype=np.int64)
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n)
                        for n in results['gt_names_3d']
                    ],
                                                       dtype=np.int64)
        results = super(DefaultFormatBundle3D, self).__call__(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str
