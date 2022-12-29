import time
import fire
import kitti_common as kitti
from mmdet3d.core.evaluation.kitti_utils.eval import  kitti_eval, kitti_eval_coco_style
from typing import List 


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(label_path,
             result_path,
             label_split_file,
             current_class:List=0,
             coco=False,
             score_thresh=-1,
             out_dir=None):
    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    if coco:
        ap_result_str, ap_dict = kitti_eval_coco_style(gt_annos, dt_annos, current_class)
        print(ap_result_str)
    else:
        ap_result_str, ap_dict = kitti_eval(gt_annos, dt_annos, current_class, out_dir=out_dir)
        print(ap_result_str)


if __name__ == '__main__':
    fire.Fire()
