

import argparse 
from mmdet3d.datasets.kitti_dataset import KittiDataset
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
import torch 
import numpy as np
from pathlib import Path 
from mmcv import Config, DictAction


def parse_results(result_dir, classes, relative_offset=0, roty_offset=0):
    results = []
    for file in sorted(list(Path(result_dir).iterdir())):
        results.append(parse_result(file, classes, relative_offset, roty_offset))
    return results

def parse_result(result_file, classes, relative_offset=0, roty_offset=0):
    result = {
        'labels_3d': [],
        'scores_3d': [],
        'boxes_3d': [],
        'boxes': [],
        'file': result_file
    }
    for line in open(result_file, 'r'):
        items = line.split(' ')
        if len(items) <= 1: continue 
        class_name = items[0].lower()
        result['labels_3d'].append(classes.index(class_name))
        result['scores_3d'].append(float(items[-1]))
        result['boxes'].append([float(v) for v in items[4:8]])
        result['boxes_3d'].append([float(items[ind]) for ind in [11, 12, 13, 10, 8, 9, 14]])
    result['labels_3d'] = torch.tensor(result['labels_3d'])
    result['scores_3d'] = torch.tensor(result['scores_3d'])
    result['boxes'] = torch.tensor(result['boxes'])
    result['boxes_3d'] = CameraInstance3DBoxes(torch.tensor(result['boxes_3d']))
    tensor = result['boxes_3d'].tensor
    tensor[:, :3].add_(tensor[:, 3:6] * relative_offset)
    tensor[:, -1].add_(roty_offset)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    parser.add_argument('--data_root')
    parser.add_argument('--split')
    parser.add_argument('--out_dir')
    args = parser.parse_args()   
    dataset = KittiDataset(args.data_root, 
            ann_file=args.data_root + '/kitti_infos_val.pkl',
            split=args.split,
            modality=dict(use_lidar=True, use_camera=True),
            classes=('car', 'pedestrian', 'cyclist'),
            test_mode=False,
            box_type_3d='camera'
    )
    results = parse_results(
        args.dir, classes=dataset.CLASSES
    )
    dataset.evaluate(results, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
