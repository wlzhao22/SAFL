import mmcv
import numpy as np
import os.path as osp

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
import pycocotools.mask as maskUtils
from mmdet.core import BitmapMasks, PolygonMasks
# from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile

PIPELINES._module_dict.pop('LoadImageFromFile')
PIPELINES._module_dict.pop('LoadAnnotations')

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 with_mixup=False,
                 with_reid=False):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.with_mixup = with_mixup
        self.with_reid = with_reid

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img',]

        if self.with_mixup:
            if results['img_prefix'] is not None:
                filename_b = osp.join(results['img_prefix'],
                                      results['img_info_b']['filename'])
            else:
                filename_b = results['img_info_b']['filename']

            img_bytes_b = self.file_client.get(filename_b)
            img_b = mmcv.imfrombytes(img_bytes_b, flag=self.color_type)

            if self.to_float32:
                img_b = img_b.astype(np.float32)

            results['filename_b'] = filename_b
            results['ori_filename_b'] = results['img_info_b']['filename']
            results['img_b'] = img_b
            results['img_shape_b'] = img_b.shape
            results['ori_shape_b'] = img_b.shape

        if self.with_reid:
            if results['img_prefix'] is not None:
                filename_c = osp.join(results['img_prefix'],
                                      results['img_info_c']['filename'])
            else:
                filename_c = results['img_info_c']['filename']

            img_bytes_c = self.file_client.get(filename_c)
            img_c = mmcv.imfrombytes(img_bytes_c, flag=self.color_type)

            if self.to_float32:
                img_c = img_c.astype(np.float32)

            results['filename_c'] = filename_c
            results['ori_filename_c'] = results['img_info_c']['filename']
            results['img_c'] = img_c
            results['img_shape_c'] = img_c.shape
            results['ori_shape_c'] = img_c.shape
            results['img_fields'].append('img_c')

        if results['img_info'].get('position', None):
            results['position'] = 0 if results['img_info']['position'] in ['f', 'r'] else 1

        if results['img_prefix'] is not None and results['img_info'].get('filename_pre', None) is not None:
            filename_pre = osp.join(results['img_prefix'],
                                    results['img_info']['filename_pre'])
            filename_post = osp.join(results['img_prefix'],
                                     results['img_info']['filename_post'])

            img_pre_bytes = self.file_client.get(filename_pre)
            img_pre = mmcv.imfrombytes(img_pre_bytes, flag=self.color_type)
            if self.to_float32:
                img_pre = img_pre.astype(np.float32)

            img_post_bytes = self.file_client.get(filename_post)
            img_post = mmcv.imfrombytes(img_post_bytes, flag=self.color_type)
            if self.to_float32:
                img_post = img_post.astype(np.float32)

            # calib_name = results['img_info']['filename'].split('.')[0] + '.txt'
            # calib_path = osp.join(results['calib_prefix'], calib_name)
            # calib = np.loadtxt(calib_path)
            img_name = results['img_info']['filename']
            calib_name = img_name.split('_')[0] + '.txt'
            # calib_name = img_name.split('.')[0] + '.txt'
            filename = osp.join(results['calib_prefix'], calib_name)
            with open(filename, 'r') as r:
                content = r.readlines()
                calib = content[2]
                calib = calib.split(' ')[1:13]
                calib = np.array([calib], dtype=np.float).reshape(3, 4)
            results['calib'] = calib
            results['inv_calib'] = np.array(np.linalg.pinv(calib), dtype=np.float)

            results["scaled_calib"] = calib
            results["scaled_calib"][:3, :] = results["scaled_calib"][:3, :] / 2
            results['scaled_inv_calib'] = np.array(np.linalg.pinv(results['scaled_calib']), dtype=np.float)

            results['scaled_calib_f'] = np.array(results["scaled_calib"], dtype=np.float)
            results['scaled_calib_f'][:3, :] = results['scaled_calib_f'][:3, :] / 4
            results['scaled_inv_calib_f'] = np.array(np.linalg.pinv(results['scaled_calib_f']), dtype=np.float)

            results['img_aug'] = img.copy()
            results['filename_pre'] = filename_pre
            results['filename_post'] = filename_post
            results['img_pre'] = img_pre
            results['img_post'] = img_post
            results['img_fields'] = ['img', 'img_aug', 'img_pre', 'img_post']
        
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in \
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam_intrinsic'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int): The max possible cat_id in input segmentation mask.
            Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids. \
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points. \
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points


    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load mutiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox2d=False,
                 with_bbox3d=False,
                 with_label=False,
                 with_reid=False,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=False,
                 file_client_args=dict(backend='disk'),
                 with_depth=False,
                 with_car_mask=False,
                 with_free_space=False,
                 with_laneSeg=False,
                 with_wheel=False,
                 with_ddd=False,
                 with_mixup=False,
                 with_affine=False, 
                 with_occluded=False,):
        self.with_bbox2d = with_bbox2d
        self.with_bbox3d = with_bbox3d
        self.with_reid = with_reid
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.with_depth = with_depth
        self.with_car_mask = with_car_mask
        self.with_free_space = with_free_space
        self.with_laneSeg = with_laneSeg
        self.with_wheel = with_wheel
        self.with_ddd = with_ddd
        self.with_mixup = with_mixup
        self.with_affine = with_affine
        self.with_occluded = with_occluded

    def _load_wheel(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_wheels'] = ann_info['wheels'].copy()
        results['gt_wheels_exist'] = ann_info['wheels_exist'].copy()

        # results['gt_wheels'][:, :2] *= results['gt_wheels_exist'][:, 0].reshape(-1, 1)
        # results['gt_wheels'][:, 2:4] *= results['gt_wheels_exist'][:, 1].reshape(-1, 1)
        # results['gt_wheels'][:, 4:6] *= results['gt_wheels_exist'][:, 2].reshape(-1, 1)
        # results['gt_wheels'][:, 6:8] *= results['gt_wheels_exist'][:, 3].reshape(-1, 1)
        for i in range(results['gt_wheels'].shape[0]):
            for j in range(4):
                if results['gt_wheels'][i, j * 2] > results['img_info']['width']:
                    results['gt_wheels'][i, j * 2] = results['img_info']['width'] - 1
                if results['gt_wheels'][i, j * 2 + 1] > results['img_info']['height']:
                    results['gt_wheels'][i, j * 2 + 1] = results['img_info']['height'] - 1
        results['wheel_fields'].append('gt_wheels')
        results['wheel_fields'].append('gt_wheels_exist')
        return results

    def _load_bboxes_2d(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes_2d'] = ann_info['bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes_2d')

        # Todo
        # ann_info_c = results['ann_info_c']
        # results['gt_bboxes_2d_c'] = ann_info_c['bboxes_2d'].copy()
        # results['bbox_fields'].append('gt_bboxes_2d_c')
        return results

    def _load_occluded(self, results):
        ann_info = results['ann_info']
        results['gt_occluded'] = ann_info['occluded'].copy()
        return results

    def _load_bboxes_2d_b(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info_b']
        results['gt_bboxes_2d_b'] = ann_info['bboxes_2d'].copy()

        # gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        # if gt_bboxes_ignore is not None:
        #     results['gt_bboxes_ignore_b'] = gt_bboxes_ignore.copy()
        #     results['bbox_fields_b'].append('gt_bboxes_ignore_b')
        # results['bbox_fields_b'].append('gt_bboxes_2d_b')
        return results

    def _load_bboxes_3d(self, results):

        ann_info = results['ann_info']
        # bboxes_3d = ann_info['bboxes_3d'].copy()
        # # gt_project_center =
        # results['gt_dims'] = bboxes_3d[:,3:]
        # results['gt_alpha'] = bboxes_3d[:,-1]
        results['gt_bboxes_3d'] = ann_info['gt_bboxes_3d'].clone()

        return results

    def _load_ddd(self, results):
        ann_info = results['ann_info']

        results['gt_ddds_head_direction'] = ann_info['ddds_head_direction'].copy()
        results['gt_ddds_dx'] = ann_info['ddds_dx'].copy()
        results['gt_ddds_dw'] = ann_info['ddds_dw'].copy()
        results['gt_ddds_l0'] = ann_info['ddds_l0'].copy()
        results['gt_ddds_l1'] = ann_info['ddds_l1'].copy()
        results['gt_ddds_l2'] = ann_info['ddds_l2'].copy()
        # results['gt_ddds_res_depth'] = ann_info['ddds_res_depth'].copy()
        results['gt_ddds_rotation'] = ann_info['ddds_rotation'].copy()
        results['gt_ddds_size'] = ann_info['ddds_size'].copy()
        results['gt_ddds_center_2d'] = ann_info['ddds_center_2d'].copy()

        results['ddd_fields'].append('gt_ddds_head_direction')
        results['ddd_fields'].append('gt_ddds_dx')
        results['ddd_fields'].append('gt_ddds_dw')
        results['ddd_fields'].append('gt_ddds_l0')
        results['ddd_fields'].append('gt_ddds_l1')
        results['ddd_fields'].append('gt_ddds_l2')
        # results['ddd_fields'].append('gt_ddds_res_depth')
        results['ddd_fields'].append('gt_ddds_rotation')
        results['ddd_fields'].append('gt_ddds_size')
        results['ddd_fields'].append('gt_ddds_center_2d')
        return results

    def _load_ddd_b(self, results):
        ann_info = results['ann_info_b']

        results['gt_ddds_head_direction_b'] = ann_info['ddds_head_direction'].copy()
        results['gt_ddds_dx_b'] = ann_info['ddds_dx'].copy()
        results['gt_ddds_dw_b'] = ann_info['ddds_dw'].copy()
        results['gt_ddds_l0_b'] = ann_info['ddds_l0'].copy()
        results['gt_ddds_l1_b'] = ann_info['ddds_l1'].copy()
        results['gt_ddds_l2_b'] = ann_info['ddds_l2'].copy()
        # results['gt_ddds_res_depth_b'] = ann_info['ddds_res_depth'].copy()
        results['gt_ddds_rotation_b'] = ann_info['ddds_rotation'].copy()
        results['gt_ddds_size_b'] = ann_info['ddds_size'].copy()
        results['gt_ddds_center_2d_b'] = ann_info['ddds_center_2d'].copy()

        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()

        # Todo
        # results['gt_labels_c'] = results['ann_info_c']['labels'].copy()
        return results

    def _load_labels_b(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels_b'] = results['ann_info_b']['labels'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)

        results['gt_lane_exist'] = results['ann_info']['lane_exist']

        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()

        results['seg_fields'].append('gt_semantic_seg')

        return results

    def _load_free_space(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # filename = results['ann_info']['mask_path']
        # class_values = results['ann_info']['class_values']

        if results['free_space_prefix'] is not None:
            filename = osp.join(results['free_space_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']

        # filename = osp.join(results['seg_prefix'],
        #                     results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        label_img = mmcv.imfrombytes(img_bytes, flag='unchanged')

        results['gt_free_space'] = label_img
        results['seg_fields'].append('gt_free_space')

        results['gt_free_space_resized'] = label_img
        results['seg_fields_resized'].append('gt_free_space_resized')

        return results

    def _load_lane_seg(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['lane_seg_prefix'], results['ann_info']['lane_seg_map'])

        img_bytes = self.file_client.get(filename)

        results['gt_lane_exist'] = results['ann_info']['lane_exist'].copy()
        results['gt_lane_class'] = results['ann_info']['lane_class'].copy()

        results['gt_lane_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged')

        results['seg_fields'].append('gt_lane_seg')
        results['lane_fields'].append('gt_lane_exist')
        results['lane_fields'].append('gt_lane_class')
        return results

    def _load_affine(self, results):
        # results['corner_pts'] = results['img_info']['corner_pts'].copy()
        results['img_affine'] = results['img'].copy()
        results['img_fields'].append('img_affine')
        # results['affine_fields'].append('corner_pts')
        return results

    def _load_depth(self, results):
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['depth_prefix'],
                            results['ann_info']['depth_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_depth'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['depth_fields'].append('gt_depth')
        return results

    def _load_reid(self, results):
        """Private function to load instance id annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded instance annotations.
        """

        results['gt_reids'] = results['ann_info']['reids'].copy()
        results['gt_reids_c'] = results['ann_info_c']['reids'].copy()
        return results

    def _load_reid_b(self, results):
        """Private function to load instance id annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded instance annotations.
        """
        results['gt_reids_b'] = results['ann_info_b']['reids'].copy()
        return results

    def _load_car_mask(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['car_mask_prefix'],
                            results['ann_info']['car_mask'])
        car_mask_bytes = self.file_client.get(filename)
        results['car_mask'] = mmcv.imfrombytes(car_mask_bytes, flag="unchanged").squeeze()
        results['car_mask_fields'].append('car_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox2d:
            results = self._load_bboxes_2d(results)
            if results is None:
                return None
        if self.with_bbox3d:
            results = self._load_bboxes_3d(results)
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        if self.with_depth:
            results = self._load_depth(results)
        if self.with_reid:
            results = self._load_reid(results)
        if self.with_car_mask:
            results = self._load_car_mask(results)
        if self.with_free_space:
            results = self._load_free_space(results)
        if self.with_laneSeg:
            results = self._load_lane_seg(results)
        if self.with_wheel:
            results = self._load_wheel(results)
        if self.with_ddd:
            results = self._load_ddd(results)
        if self.with_affine:
            results = self._load_affine(results)

        if self.with_mixup:
            if self.with_bbox2d:
                results = self._load_bboxes_2d_b(results)
            if self.with_label:
                results = self._load_labels_b(results)
            if self.with_ddd:
                results = self._load_ddd_b(results)
            if self.with_reid:
                results = self._load_reid_b(results)

        if self.with_occluded: 
            results = self._load_occluded(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox2d={self.with_bbox2d}, '
        repr_str += f'(with_bbox3d={self.with_bbox3d}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'with_reid={self.with_reid}, '
        repr_str += f'with_depth={self.with_depth})'
        repr_str += f'with_calib={self.with_calib})'
        repr_str += f'with_free_space={self.free_space})'
        repr_str += f'with_laneSeg={self.with_laneSeg}, '
        repr_str += f'with_wheel={self.with_wheel}, '
        repr_str += f'poly2mask={self.poly2mask})'
        repr_str += f'poly2mask={self.file_client_args})'
        repr_str += f'with_ddd={self.with_ddd}, '

        return repr_str

@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 with_occluded=False,
                 poly2mask=True,
                 seg_3d_dtype='int',
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox2d=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            with_occluded=with_occluded,
            poly2mask=poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d'].clone()
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.long)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.long)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str
