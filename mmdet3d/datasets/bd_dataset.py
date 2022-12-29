'''
Author: your name
Date: 2020-12-07 10:17:01
LastEditTime: 2021-06-28 17:58:00
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /BDPilot/mmdet/datasets/bd_dataset.py
'''
'''=================================================
               ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃              ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━-┓
                ┃Beast god bless┣┓
                ┃　Never BUG ！ ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
=================================================='''

import os.path as osp
import os

import mmcv
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose
import random


@DATASETS.register_module()
class BDDataset(Dataset):
    
    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix=None,
                 seg_prefix=None,
                 calib_prefix=None,
                 free_space_prefix=None,
                 lane_seg_prefix=None,
                 depth_prefix=None,
                 car_mask_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 with_wheel =False,
                 with_ddd=False,
                 with_reid=False,
                 reid_range=3,
                 filter_empty_gt=True):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.free_space_prefix = free_space_prefix
        self.lane_seg_prefix = lane_seg_prefix
        self.calib_prefix = calib_prefix
        self.depth_prefix = depth_prefix
        self.car_mask_prefix = car_mask_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)
        self.with_wheel = with_wheel
        self.with_ddd = with_ddd
        self.reid_range=reid_range
        self.with_reid = with_reid

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.depth_prefix is None or osp.isabs(self.depth_prefix)):
                self.depth_prefix = osp.join(self.data_root, self.depth_prefix)
            if not (self.lane_seg_prefix is None or osp.isabs(self.lane_seg_prefix)):
                self.lane_seg_prefix = osp.join(self.data_root, self.lane_seg_prefix)
            if not (self.car_mask_prefix is None or osp.isabs(self.car_mask_prefix)):
                self.car_mask_prefix = osp.join(self.data_root, self.car_mask_prefix)
            if not (self.calib_prefix is None or osp.isabs(self.calib_prefix)):
                self.calib_prefix = osp.join(self.data_root, self.calib_prefix)
            if not (self.free_space_prefix is None or osp.isabs(self.free_space_prefix)):
                self.free_space_prefix = osp.join(self.data_root, self.free_space_prefix)
            if not (self.proposal_file is None or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root, self.proposal_file)
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)
        # filter data infos if classes are customized
        if self.custom_classes:
            self.data_infos = self.get_subset_by_classes()

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
            
        # filter images too small
        # if not test_mode:
        #     valid_inds = self._filter_imgs()
        #     self.data_infos = [self.data_infos[i] for i in valid_inds]
        #     if self.proposals is not None:
        #         self.proposals = [self.proposals[i] for i in valid_inds]

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            if 'filename_pre' in info:
                info['filename_pre'] = info['file_name_pre']
                info['filename_post'] = info['file_name_post']
            data_infos.append(info)
        
        return data_infos

    def load_proposals(self, proposal_file):
        """Load proposal from proposal file."""
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes_2d = []
        gt_bboxes_3d = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_ids = []
        gt_wheels = []
        gt_wheels_exist = []

        gt_lane_exist = []
        gt_lane_class = []

        gt_ddds_head_direction = [] #[0, 0, 0, 0] left, right, front, rear
        gt_ddds_dx = [] # 2d center res
        gt_ddds_dw = [] # 2d face width
        gt_ddds_l0 = [] # rear/front edge height 0
        gt_ddds_l1 = [] # rear/front edge height 1
        gt_ddds_l2 = [] # rear/front edge height 2
        gt_ddds_res_depth = [] # res depth
        gt_ddds_rotation = []
        gt_ddds_size = []
        gt_ddds_center_2d = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox_2d']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox_2d = [x1, y1, x1 + w, y1 + h]
            # if x1 + w/2 <= 0 or y1 + h/2 <= 0:
            #     continue
            if 'bbox_3d' in ann:
                bbox_3d = ann['bbox_3d']
            else:
                bbox_3d = []
            if 'instance_id' in ann:
                id = ann['instance_id']
            else:
                id = -1
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox_2d)
            else:
                gt_bboxes_2d.append(bbox_2d)
                gt_bboxes_3d.append(bbox_3d)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])
                gt_ids.append(id)
                if self.with_wheel == True:
                    gt_wheels.append(ann['wheel'])
                    gt_wheels_exist.append(ann['wheel_exist'])
                
                if self.with_ddd:
                    gt_ddds_head_direction.append(ann['head_direction'])
                    gt_ddds_dx.append(ann['dx'])
                    gt_ddds_dw.append(ann['dw'])
                    gt_ddds_l0.append(ann['l0'])
                    gt_ddds_l1.append(ann['l1'])
                    gt_ddds_l2.append(ann['l2'])
                    gt_ddds_res_depth.append(ann['res_depth'])
                    gt_ddds_rotation.append(ann['rotation'])
                    gt_ddds_size.append(ann['size'])
                    gt_ddds_center_2d.append(ann['center_2d'])

        if gt_bboxes_2d:
            gt_bboxes_2d = np.array(gt_bboxes_2d, dtype=np.float32)
            gt_bboxes_3d = np.array(gt_bboxes_3d,dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_ids = np.array(gt_ids,dtype=np.int64)
            if self.with_wheel:
                gt_wheels = np.array(gt_wheels, dtype=np.int64)
                gt_wheels_exist = np.array(gt_wheels_exist, dtype=np.int64)
            
            if self.with_ddd:
                gt_ddds_head_direction = np.array(gt_ddds_head_direction, dtype=np.int64)
                gt_ddds_dx = np.array(gt_ddds_dx, dtype=np.float32)
                gt_ddds_dw = np.array(gt_ddds_dw, dtype=np.float32)
                gt_ddds_l0 = np.array(gt_ddds_l0, dtype=np.float32)
                gt_ddds_l1 = np.array(gt_ddds_l1, dtype=np.float32)
                gt_ddds_l2 = np.array(gt_ddds_l2, dtype=np.float32)
                gt_ddds_res_depth = np.array(gt_ddds_res_depth, dtype=np.float32)
                gt_ddds_rotation = np.array(gt_ddds_rotation, dtype=np.float32)
                gt_ddds_size = np.array(gt_ddds_size, dtype=np.float32)
                gt_ddds_center_2d = np.array(gt_ddds_center_2d, dtype=np.float32)
        else:
            gt_bboxes_2d = np.zeros((0, 4), dtype=np.float32)
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_ids = np.array([], dtype=np.int64)
            if self.with_wheel:
                gt_wheels = np.array([], dtype=np.int64)
                gt_wheels_exist = np.array([], dtype=np.int64)
            
            if self.with_ddd:
                gt_ddds_head_direction = np.array([], dtype=np.int64)
                gt_ddds_dx = np.array([], dtype=np.float32)
                gt_ddds_dw = np.array([], dtype=np.float32)
                gt_ddds_l0 = np.array([], dtype=np.float32)
                gt_ddds_l1 = np.array([], dtype=np.float32)
                gt_ddds_l2 = np.array([], dtype=np.float32)
                gt_ddds_res_depth = np.array([], dtype=np.float32)
                gt_ddds_rotation = np.array([], dtype=np.float32)
                gt_ddds_size = np.array([], dtype=np.float32)
                gt_ddds_center_2d = np.array([], dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')
        
        # seg_map = img_info['filename']

        if 'lane_poly' in img_info:
            lane_poly = np.array(img_info['lane_poly'])
        else:
            lane_poly = None

        # lane_seg_map = img_info['filename'].replace('jpg', 'png')
        # lane_seg_map = lane_seg_map.replace('frame','lane')
        if 'lanes_exist' in img_info:
            lane_seg_map = img_info['filename'].replace('jpg', 'png')
            # lane_seg_map = lane_seg_map.replace('frame','lane')
            lane_exist = np.array(img_info['lanes_exist'])
            lane_class = np.array(img_info['lanes_class'])
        else:
            lane_seg_map = None
            lane_exist = None
            lane_class = None


        ann = dict(
            bboxes_2d=gt_bboxes_2d,
            bboxes_3d=gt_bboxes_3d,
            reids = gt_ids,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            lane_seg_map = lane_seg_map,
            lane_exist = lane_exist,
            lane_class = lane_class,
            lane_poly = lane_poly,
            wheels=gt_wheels,
            wheels_exist=gt_wheels_exist,
            ddds_head_direction=gt_ddds_head_direction,
            ddds_dx = gt_ddds_dx,
            ddds_dw = gt_ddds_dw,
            ddds_l0 = gt_ddds_l0,
            ddds_l1 = gt_ddds_l1,
            ddds_l2 = gt_ddds_l2,
            ddds_res_depth = gt_ddds_res_depth,
            ddds_rotation = gt_ddds_rotation,
            ddds_size = gt_ddds_size,
            ddds_center_2d = gt_ddds_center_2d,
            car_mask="mask.png",)

        return ann

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['calib_prefix'] = self.calib_prefix
        results['free_space_prefix'] = self.free_space_prefix
        results['lane_seg_prefix'] = self.lane_seg_prefix
        results['depth_prefix'] = self.depth_prefix
        results['car_mask_prefix'] = self.car_mask_prefix
        results['lane_fields'] = []
        results['polylane_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['depth_fields'] = []
        results['img_fields'] = []
        results['car_mask_fields'] = []
        results['wheel_fields'] = []
        results['ddd_fields'] = []
        results['affine_fields'] = []
        results['seg_fields_resized'] = []


    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)

        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
    
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        selected_idx = idx
        while selected_idx != idx:
            selected_idx = random.randint(0, len(self.data_infos)-1)
        img_info_b = self.data_infos[selected_idx]
        ann_info_b = self.get_ann_info(selected_idx)

        file_id = int(os.path.basename(img_info['filename']).split('.')[0].split('_')[-1])
        left_range, right_range = -self.reid_range, self.reid_range
        
        if self.with_reid:
            while 1:
                res_idx = random.randint(left_range, right_range)
                if idx+res_idx < 0:
                    left_range = res_idx + 1
                    continue
                elif idx + res_idx >= len(self.data_infos):
                    right_range = res_idx - 1
                    continue
                
                img_info_c = self.data_infos[idx+res_idx]
                ann_info_c = self.get_ann_info(idx+res_idx)
                file_id_c = int(os.path.basename(img_info_c['filename']).split('.')[0].split('_')[-1])
                if file_id + res_idx == file_id_c:
                    break
                elif res_idx > 0:
                    right_range = res_idx - 1
                else:
                    left_range = res_idx + 1    
            results = dict(img_info=img_info, ann_info=ann_info, img_info_b=img_info_b, ann_info_b=ann_info_b, img_info_c=img_info_c, ann_info_c=ann_info_c)
        else:
            results = dict(img_info=img_info, ann_info=ann_info, img_info_b=img_info_b, ann_info_b=ann_info_b)
       
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
        """
        if classes is None:
            cls.custom_classes = False
            return cls.CLASSES

        cls.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def get_subset_by_classes(self):
        """Get img ids that contain any category in class_ids.

        Different from the coco.getImgIds(), this function returns the id if
        the img contains one of the categories rather than all.

        Args:
            class_ids (list[int]): list of category ids

        Return:
            ids (list[int]): integer list of img ids
        """

        ids = set()
        for i, class_id in enumerate(self.cat_ids):
            ids |= set(self.coco.catToImgs[class_id])
        self.img_ids = list(ids)

        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
