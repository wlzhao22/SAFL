import mmcv
import numpy as np
import os.path as osp
from pathlib import Path 

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
import pycocotools.mask as maskUtils
from mmdet.core import BitmapMasks, PolygonMasks

from mmdet.datasets.pipelines.loading import LoadImageFromFile

from mmdet3d.datasets.pipelines.loading import LoadAnnotations3D

@PIPELINES.register_module()
class LoadImageFromFileMonoFlex(LoadImageFromFile):
    def __init__(self, to_float32=False, color_type='color', with_right_image=False, n_prevs:int=0, file_client_args=dict(backend='disk')):
        super().__init__(to_float32=to_float32, color_type=color_type, file_client_args=file_client_args)
        self.n_prevs = n_prevs
        self.with_right_image = with_right_image

    def __call__(self, results):
        results = super().__call__(results)
        h, w, ch = results['img'].shape

        for i in range(self.n_prevs):
            if results['prev_prefix'] is not None: 
                filename = osp.join(results['prev_prefix'], results['img_info']['prev_image_template'].format(i + 1))
            else:
                filename = results['img_info']['prev_image_template'].format(i + 1)
            
            if not osp.isfile(filename):
                img = np.zeros((h, w, ch), np.uint8)
            else:
                img_bytes = self.file_client.get(filename)
                img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
            if self.to_float32:
                img = img.astype(np.float32)
            results['filename_prev{}'.format(i + 1)] = filename
            results['ori_filename_prev{}'.format(i + 1)] = results['img_info']['filename']
            results['img_prev{}'.format(i + 1)] = img 
            results['img_shape_prev{}'.format(i + 1)] = img.shape 
            results['ori_shape_prev{}'.format(i + 1)] = img.shape 
            results['img_fields'].append('img_prev{}'.format(i + 1))

        if self.with_right_image:
            img_left_filename = Path(results['img_info']['filename'])
            img_right_filename = img_left_filename.parent.parent / 'image_3' / img_left_filename.name 
            img_bytes = self.file_client.get(img_right_filename)
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
            if self.to_float32:
                img = img.astype(np.float32)
            results['img_r'] = img 
            results['img_shape_r'] = img.shape
            results['ori_shape_r'] = img.shape
            results['img_fields'].append('img_r')

        return results 


@PIPELINES.register_module()
class LoadAnnotations3DMonoFlex(LoadAnnotations3D):
    def __init__(self, with_bbox_3d=True, with_label_3d=True, with_attr_label=False, with_mask_3d=False, with_seg_3d=False, with_bbox=False, with_label=False, with_mask=False, with_seg=False, with_bbox_depth=False, with_occluded=False, 
    with_flow=False, poly2mask=True, seg_3d_dtype='int', file_client_args=...):
        super().__init__(with_bbox_3d=with_bbox_3d, with_label_3d=with_label_3d, with_attr_label=with_attr_label, with_mask_3d=with_mask_3d, with_seg_3d=with_seg_3d, with_bbox=with_bbox, with_label=with_label, with_mask=with_mask, with_seg=with_seg, with_bbox_depth=with_bbox_depth, with_occluded=with_occluded, poly2mask=poly2mask, seg_3d_dtype=seg_3d_dtype, file_client_args=file_client_args)
        self.with_flow = with_flow 

    def _load_flow(self, results):
        img_filename = Path(results['img_info']['filename'])
        flow_dir = img_filename.parent.parent / 'flow_2_flownet3'
        h, w, _ = results['img_shape']
        list_flows = []
        list_available = []
        flow_fields = []
        for i in range(3):
            filename = flow_dir / '{}_{:02}.npy'.format(img_filename.stem, i + 1)

            if not osp.isfile(filename):
                flow = np.zeros((h, w, 2), np.float32)
                available = False 
            else:
                flow = np.load(str(filename), allow_pickle=True)[0].transpose(1, 2, 0)[:h, :w, :]
                available = True 

            key = 'flow_{}'.format(i + 1)
            results[key] = flow
            flow_fields.append(key)
            list_available.append(available)

        results['flow_fields'] = flow_fields
        results['flow_available'] = list_available
        return results

    def __call__(self, results):
        results = super().__call__(results)
        if self.with_flow:
            results = self._load_flow(results)
        return results 
