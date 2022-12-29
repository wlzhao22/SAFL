import argparse
import os
import json 
import pandas as pd 
from collections import defaultdict

import matplotlib.pyplot as plt 

import mmcv
import torch
from torch import Tensor
import numpy as np
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.parallel import DataContainer as DC

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector

from mmdet3d.models.utils.utils_2d.instances import Instances
from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
from mmdet3d.core.bbox.structures.cam_box3d import CameraInstance3DBoxes

from mmdet3d.core import box3d_multiclass_nms, xywhr2xyxyr

from types import SimpleNamespace
from itertools import chain 
import cv2 as cv 
from tqdm import tqdm 
from pathlib import Path 
import re 
from tools.data_converter.kitti_raw_converter import get_all_image_ids
from argparse import ArgumentParser
data_root = 'data/kitti'
import pickle 

pipeline = [
    dict(type='LoadImageFromFile'), 
    dict(type='Collect3D', keys=['img'])
]
class_names = ["Car", "Pedestrian", "Cyclist"]

input_modality = dict(use_lidar=True, use_camera=True)

data = dict(
    samples_per_gpu=1, 
    workers_per_gpu=0, 
    train=dict(
        type='KittiDataset', 
        data_root=data_root, 
        ann_file=data_root + '/kitti_infos_train.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='camera'
    ),
    val=dict(
        type='KittiDataset',
        data_root=data_root,
        ann_file=data_root + '/kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='camera'
    ),
    test=dict(
        type='KittiDataset',
        data_root=data_root,
        ann_file=data_root + '/kitti_infos_test.pkl',
        split='testing',
        pts_prefix='velodyne_reduced',
        pipeline=pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='camera'
    )    
)


def main():
    parser = ArgumentParser()
    parser.add_argument('--kitti_raw_data_root', default='data/kitti-raw')
    parser.add_argument('--thresh', default=0.98)
    parser.add_argument('--output_dir', required=False)
    parser.add_argument('--bins', default=60)
    parser.add_argument('--percent_train', default=0.5)
    args = parser.parse_args()

    if args.output_dir is None: 
        output_dir = args.kitti_raw_data_root 
    else:
        output_dir = args.output_dir 
    output_dir = Path(output_dir)
    

    global data 
    dataset_val = build_dataset(data['val'])
    dataset_test = build_dataset(data['test'])
    print('len dataset val: ', len(dataset_val))
    print('len dataset test: ', len(dataset_test))

    list_hist = []
    def get_hist(img):
        # img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # hue = img_hsv[..., 0]
        # hist, bin_eddges = np.histogram(hue, bins=args.bins)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        hist, bin_edges = np.histogram(img_gray, bins=args.bins)
        return hist 

    print("Scanning the val split of KITTI detection dataset.")
    len_dataset_val = len(dataset_val) 
    for i in tqdm(range(len_dataset_val)):
        list_hist.append(get_hist(dataset_val[i]['img']))
    print("Scanning the testing split of KITTI detection dataset.")
    len_dataset_test = len(dataset_test)
    for i in tqdm(range(len_dataset_test)):
        list_hist.append(get_hist(dataset_test[i]['img']))

    hist_mat = np.stack(list_hist)  # shape == (11287, bins) 
    del list_hist
    hist_mat = hist_mat / np.sqrt(np.sum(hist_mat**2, -1, keepdims=True))
    
    img_ids_raw = get_all_image_ids(args.kitti_raw_data_root)
    all_records = []
    for i, img_info in enumerate(tqdm(img_ids_raw)):
        img = cv.imread(str(img_info['image02_path']), cv.IMREAD_COLOR)
        h = get_hist(img) # shape == (bins, )
        h = h / np.sqrt(np.sum(h**2))
        sim_matrix = np.sum(h * hist_mat, -1)
        sim = np.max(sim_matrix) 
        record = img_info.copy() 
        record.update({
            'similarity': sim 
        })
        identic = False 
        if sim >= args.thresh: 
            for idx in np.nonzero(sim_matrix == sim)[0]:
                if idx < len(dataset_val):
                    path1 = dataset_val[idx]['img_metas'].data['filename']
                else:
                    path1 = dataset_test[idx - len_dataset_val]['img_metas'].data['filename']
                path2 = img_info['image02_path']
                # print('Similar: {} and {}'.format(path1, path2))
                im1 = cv.imread(path1)
                im2 = cv.imread(path2)
                if im1.shape == im2.shape and np.all(im1 == im2):
                    print('Identic images: {} and {}'.format(path1, path2))
                    identic = True 
                    break
        if not identic:
            all_records.append(record)
    all_records.sort(key = lambda x: x['similarity'])

    n_train = int(args.percent_train * len(all_records))
    train = all_records[:n_train]
    val = all_records[n_train:]

    print('{} training images and {} val images in total.'.format(len(train), len(val)))
    with open(output_dir / 'kitti_raw_split_train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(output_dir / 'kitti_raw_split_val.pkl', 'wb') as f:
        pickle.dump(val, f)
    with open(output_dir / 'kitti_raw_split_all.pkl', 'wb') as f:
        pickle.dump(all_records, f)
        
    return 


if __name__ == '__main__':
    main() 

