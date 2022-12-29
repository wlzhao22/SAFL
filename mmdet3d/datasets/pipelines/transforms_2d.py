#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
             ┏┓      ┏┓
            ┏┛┻━━━━━━┛┻┓
            ┃          ┃
            ┃  ┳┛  ┗┳  ┃
            ┃     ┻    ┃
            ┗━┓      ┏━┛
              ┃      ┗━━━━━━━-┓
              ┃Beast god bless┣┓
              ┃　Never BUG ！ ┏┛
              ┗┓┓┏━━━━━━━━┳┓┏┛
               ┃┫┫        ┃┫┫
               ┗┻┛        ┗┻┛
=================================================='''
import inspect
import cv2

import mmcv
import numpy as np
from numpy import random
import math

from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets.builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


PIPELINES._module_dict.pop('Resize')
PIPELINES._module_dict.pop('RandomFlip')
PIPELINES._module_dict.pop('Pad')
PIPELINES._module_dict.pop('Normalize')
PIPELINES._module_dict.pop('RandomCrop')
PIPELINES._module_dict.pop('SegRescale')
PIPELINES._module_dict.pop('PhotoMetricDistortion')
PIPELINES._module_dict.pop('Expand')
PIPELINES._module_dict.pop('MinIoURandomCrop')
PIPELINES._module_dict.pop('Corrupt')
PIPELINES._module_dict.pop('Albu')
PIPELINES._module_dict.pop('RandomCenterCropPad')


def rand_uniform_strong(min, max):
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min

def rand_scale(s):
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale


@PIPELINES.register_module()
class MixUp(object):
    def __init__(self,rand_thr=0.5, alpha=1.5, beta=1.5, with_ddd=False, with_reid=False):
        self.alpha = alpha
        self.beta = beta
        self.rand_thr = rand_thr
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

        self.with_ddd = with_ddd
        self.with_reid = with_reid

    def __call__(self, results):
        if np.random.uniform(0., 1.) < self.rand_thr:
            return results

        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            return results

        img_a = results['img_affine']
        gt_bbox_a = results['gt_bboxes_2d']
        img_b = results['img_b']
        gt_bbox_b = results['gt_bboxes_2d_b']
        out_img = self._mixup_img(img_a,img_b,factor)
        gt_bbox = np.concatenate([gt_bbox_a, gt_bbox_b], axis=0)

        gt_labels_a = results['gt_labels']
        gt_labels_b = results['gt_labels_b']
        gt_labels = np.concatenate((gt_labels_a, gt_labels_b), axis=0)

        if self.with_reid:
            gt_reids_a = results['gt_reids']
            gt_reids_b = results['gt_reids_b']
            gt_reids = np.concatenate((gt_reids_a, gt_reids_b), axis=0)
            results['gt_reids'] = gt_reids

        results['gt_bboxes_2d'] = gt_bbox
        results['gt_labels'] = gt_labels

        results['img_affine'] = out_img
        results['img_shape'] = out_img.shape

        if self.with_ddd:
            gt_ddds_head_direction_a = results['gt_ddds_head_direction']
            gt_ddds_head_direction_b = results['gt_ddds_head_direction_b']
            results['gt_ddds_head_direction'] = np.concatenate((gt_ddds_head_direction_a, gt_ddds_head_direction_b), axis=0)

            gt_ddds_dx_a = results['gt_ddds_dx']
            gt_ddds_dx_b = results['gt_ddds_dx_b']
            results['gt_ddds_dx'] = np.concatenate((gt_ddds_dx_a, gt_ddds_dx_b), axis=0)

            gt_ddds_dw_a = results['gt_ddds_dw']
            gt_ddds_dw_b = results['gt_ddds_dw_b']
            results['gt_ddds_dw'] = np.concatenate((gt_ddds_dw_a, gt_ddds_dw_b), axis=0)

            gt_ddds_l0_a = results['gt_ddds_l0']
            gt_ddds_l0_b = results['gt_ddds_l0_b']
            results['gt_ddds_l0'] = np.concatenate((gt_ddds_l0_a, gt_ddds_l0_b), axis=0)

            gt_ddds_l1_a = results['gt_ddds_l1']
            gt_ddds_l1_b = results['gt_ddds_l1_b']
            results['gt_ddds_l1'] = np.concatenate((gt_ddds_l1_a, gt_ddds_l1_b), axis=0)

            gt_ddds_l2_a = results['gt_ddds_l2']
            gt_ddds_l2_b = results['gt_ddds_l2_b']
            results['gt_ddds_l2'] = np.concatenate((gt_ddds_l2_a, gt_ddds_l2_b), axis=0)
            
            gt_ddds_rotation_a = results['gt_ddds_rotation']
            gt_ddds_rotation_b = results['gt_ddds_rotation_b']
            results['gt_ddds_rotation'] = np.concatenate((gt_ddds_rotation_a, gt_ddds_rotation_b), axis=0)
            
            gt_ddds_size_a = results['gt_ddds_size']
            gt_ddds_size_b = results['gt_ddds_size_b']
            results['gt_ddds_size'] = np.concatenate((gt_ddds_size_a, gt_ddds_size_b), axis=0)
            
            gt_ddds_center_2d_a = results['gt_ddds_center_2d']
            gt_ddds_center_2d_b = results['gt_ddds_center_2d_b']
            results['gt_ddds_center_2d'] = np.concatenate((gt_ddds_center_2d_a, gt_ddds_center_2d_b), axis=0)

        return results

    def _mixup_img(self, img1, img2, factor):
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')


@PIPELINES.register_module()
class Mosaic(object):
    def __init__(self,transforms=dict(),img_scale=608):
        self.img_scale = img_scale
        self.mosaic_border = [-img_scale // 2, -img_scale // 2]

        self.pipeline = Compose(transforms)

    def __call__(self, results):
        s = self.img_scale
        results['mosaic_border'] = self.mosaic_border
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic 中心坐标
        labels4 = []
        gt_classes = []
        gt_scores = []
        truth = results['truth']
        result_0 = results
        imgs = [random.choice(list(truth.keys())) for _ in range(3)]
        result_4 = [result_0] + [dict(y_true=truth[img_], img_name=img_,img_prefix=results['img_prefix']) for img_ in imgs]
        for idx,result in enumerate(result_4):
            result = self.pipeline(result)
            img, (h, w) = result['img'],result['img_shape'][:2]
            if idx == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)大图位置
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)小图位置 从右下角剪切

            elif idx == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h  # 从左下角剪切
            elif idx == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)  # 从右上角剪切
            elif idx == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)  # 从左上角剪切

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            gt_bboxes = result['gt_bboxes_2d']
            gt_class = result['gt_labels']
            labels = gt_bboxes.copy()
            if len(labels):  # Normalized xywh to pixel xyxy format
                labels[:, [0,2]] = w * labels[:, [0,2]] + padw  # pad width
                labels[:, [1,3]] = h * labels[:, [1,3]] + padh  # pad height
            labels4.append(labels)
            gt_classes.append(gt_class)
            gt_scores.append(gt_score)

        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            gt_classes = np.concatenate(gt_classes, 0)
            gt_scores = np.concatenate(gt_scores, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            np.clip(labels4, 0, 2 * s, out=labels4)  # use with random_affine

            _w = labels4[:, 2] - labels4[:, 0]
            _h = labels4[:, 3] - labels4[:, 1]
            area = _w * _h
            labels4 = labels4[np.where(area>20)]
            gt_classes = gt_classes[np.where(area>20)]
            gt_scores = gt_scores[np.where(area>20)]

            results['gt_bboxes_2d'] = labels4
            results['gt_labels'] = gt_classes

        results['img'] = img4
        results['img_shape'] = img4.shape

        return results

#随机平移
@PIPELINES.register_module()
class RandomTranslation(object):

    def __init__(self,random_thr=0.5):
        self.random_thr = random_thr

    def __call__(self, results):
        if random.random() < self.random_thr:
            return results

        img = results['img']
        gt_bboxes = results['gt_bboxes_2d']
        h,w = img.shape[:2]

        x_min,x_max,y_min,y_max = w ,0,h,0
        for bbox in gt_bboxes:
            x_min ,y_min,x_max,y_max= min(x_min, bbox[0]),min(y_min, bbox[1]),max(x_max, bbox[2]),max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # ---------------------- 平移boundingbox ----------------------
        shift_bboxes = list()
        for bbox in gt_bboxes:
            shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y])

        shift_bboxes = np.array(shift_bboxes)
        results['img'] = shift_img
        results['img_shape'] = shift_img.shape
        results['gt_bboxes_2d'] =shift_bboxes

        return results

# #随机放射变换
# @PIPELINES.register_module()
# class RandomAffineCornerPoint(object):

#     def __init__(self,degrees=10, translate=.1, scale=.5, shear=0.0, border=(0, 0), c=8):
#         self.degrees = degrees
#         self.translate = translate
#         self.scale = scale
#         self.shear = shear
#         self.border = border
#         self.c = c

#     def __call__(self, results):
#         # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
#         # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
#         # targets = [cls, xyxy]
#         img = results['img_affine']
#         org_corner_pts = results['corner_pts'].copy().reshape(-1,2)
#         targets = results['corner_pts'].copy()
#         height = img.shape[0] + self.border[0] * 2  # shape(h,w,c)
#         width = img.shape[1] + self.border[1] * 2

#         # Rotation and Scale 旋转和缩放
#         R = np.eye(3)
#         a = random.uniform(-self.degrees, self.degrees)
#         # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
#         s = random.uniform(1 - self.scale, 1 + self.scale)
#         # s = 2 ** random.uniform(-scale, scale)
#         R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

#         # Translation 平移
#         T = np.eye(3)
#         T[0, 2] = random.uniform(-self.translate, self.translate) * img.shape[1] + self.border[1]  # x translation (pixels)
#         T[1, 2] = random.uniform(-self.translate, self.translate) * img.shape[0] + self.border[0]  # y translation (pixels)

#         # Shear 剪切
#         S = np.eye(3)
#         S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
#         S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

#         # Combined rotation matrix
#         M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
#         if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
#             img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

#         # Transform label coordinates
#         n = len(targets)
#         xy_id = []
#         match_xy_pts = []
#         affine_negtive_pt = []
#         if n:
#             # warp points
#             xy = np.ones((n, 3))
#             xy[:, :2] = targets # x1y1, x2y2, x1y2, x2y1
#             xy = (xy @ M.T)[:, :2].reshape(n, 2)
#             xy_index = np.where((xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height))
#             xy_id, org_corner_pts_id = xy[xy_index], org_corner_pts[xy_index]
#             match_xy_pts = np.concatenate((org_corner_pts_id, xy_id), axis=1)        
#             cv2.imwrite("test.png", img)
#             # affine_negtive_pts = self.get_nearest_c_xy(xy_id, self.c)
        
#         # negtive_pts = self.get_nearest_c_xy(org_corner_pts, self.c)

        
#         results['img_affine'] = img
#         results['corner_pts'] = org_corner_pts
#         # results['negative_corner_pts'] = negtive_pts

#         results['corner_pts_affine'] = xy_id
#         results['match_corner_pts'] = match_xy_pts
#         # results['negative_affine_corner_pts'] = affine_negtive_pts

#         return results

@PIPELINES.register_module()
class RandomAffine(object):

    def __init__(self,degrees=10, translate=.1, scale=.5, shear=0.0, border=(0, 0), with_reid=False):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.border = border
        self.with_reid = with_reid

    def make_affine(self, img, targets, gt_class, gt_reids=None):
        height = img.shape[0] + self.border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + self.border[1] * 2
        
        # Rotation and Scale 旋转和缩放
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation 平移
        T = np.eye(3)
        T[0, 2] = random.uniform(-self.translate, self.translate) * img.shape[1] + self.border[1]  # x translation (pixels)
        T[1, 2] = random.uniform(-self.translate, self.translate) * img.shape[0] + self.border[0]  # y translation (pixels)

        # Shear 剪切
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Combined rotation matrix
        M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
        if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

        # Transform label coordinates
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            # xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # reject warped points outside of image
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            area0 = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
            i = (w > 2) & (h > 2) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 20)

            targets = targets[i]
            gt_class = gt_class[i]
            targets[:, 0:4] = xy[i]
            # if self.with_reid:
            #     gt_reids = results['gt_reids'][i]
            #     results['gt_reids'] = gt_reids
            
            # results['img_affine'] = img
            # results['gt_bboxes_2d'] = targets
            # results['gt_labels'] = gt_class
            if self.with_reid:
                gt_reids = gt_reids[i]
                return img, targets, gt_class, gt_reids
            else:
                return img, targets, gt_class

    def __call__(self, results):
        
        if self.with_reid:
            for img_key, bbox_key, label_key, id_key in zip(['img_affine','img_c'], ['gt_bboxes_2d', 'gt_bboxes_2d_c'], ['gt_labels', 'gt_labels_c'], ['gt_reids', 'gt_reids_c']):
                img = results[img_key]
                targets = results[bbox_key]
                gt_class = results[label_key]
                gt_reids = results[id_key]
                results[img_key], results[bbox_key], results[label_key], results[id_key] = self.make_affine(img, targets, gt_class, gt_reids)
        else:
            for img_key, bbox_key, label_key in zip(['img_affine'], ['gt_bboxes_2d'], ['gt_labels']):
                img = results[img_key]
                targets = results[bbox_key]
                gt_class = results[label_key]
                results[img_key], results[bbox_key], results[label_key] = self.make_affine(img, targets, gt_class)
        
        # img = results["img_affine"]
        # targets = results["gt_bboxes_2d"]
        # for bbox in targets:
        #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
        # cv2.imwrite("test.jpg", img)
        
        # img = results["img_c"]
        # targets = results["gt_bboxes_2d_c"]
        # for bbox in targets:
        #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
        # cv2.imwrite("test2.jpg", img)
        
        return results


@PIPELINES.register_module()
class RandomHSV(object):
    def __init__(self,hue=0.1,saturation=1.5,exposure=1.5,rand_thr=0.5):
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure
        self.rand_thr = rand_thr


    def __call__(self, results):
        if random.random() < self.rand_thr:
            return results
        dhue = rand_uniform_strong(-self.hue,self.hue)  # 色调
        dsat = rand_scale(self.saturation)  # 饱和度
        dexp = rand_scale(self.exposure)  # 曝光度
        if dsat != 1 or dexp != 1 or dhue != 0:
            img = results['img']
            if img.shape[2] >= 3:
                hsv_src = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
                hsv = cv2.split(hsv_src)
                hsv[1] *= dsat
                hsv[2] *= dexp
                hsv[0] += 179 * dhue
                hsv_src = cv2.merge(hsv)
                img = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)
            else:
                img *= dexp

        results['img'] = img

        return results

@PIPELINES.register_module()
class RandomBlur(object):
    def __init__(self,rand_thr=0.5):
        self.rand_thr = rand_thr

    def __call__(self, results):
        if random.random()> self.rand_thr:
            img = results['img']
            dst = cv2.GaussianBlur(img, (3, 3), 0)
            results['img'] = dst
            results['img_shape'] = dst.shape
        return results

@PIPELINES.register_module()
class RandomNoise(object):
    def __init__(self,gaussian_noise=50,rand_thr=0.5):
        self.rand_thr = rand_thr
        self.gaussian_noise = gaussian_noise

    def __call__(self, results):
        if random.random()> self.rand_thr:
            img = results['img']
            gaussian_noise = min(self.gaussian_noise, 127)
            gaussian_noise = max(gaussian_noise, 0)
            gaussian_noise = random.randint(0, gaussian_noise)
            noise = np.random.normal(0, gaussian_noise, img.shape)
            img = img + noise
            results['img'] = img
            results['img_shape'] = img.shape

        return results

@PIPELINES.register_module()
class RandomRotate(object):
    """Rotate images & labels"""

    def __init__(self, angle):
        self.angle = angle
    
    def __call__(self, results):

        angle = random.randint(0, self.angle*2) - self.angle

        for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imrotate(
                    results[key], angle)

        for key in results.get('seg_fields', ['gt_semantic_seg']):
                results[key] = mmcv.imrotate(
                    results[key], angle)

        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(angle={self.angle}, '
        
@PIPELINES.register_module()
class Resize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 backend='cv2'):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img_affine(self, results):
        if results.get('affine_fields', None) is not None:
            for key in results.get('affine_fields', None):
                if key == "corner_pts":
                    corner_pts = results[key] * results['scale_factor'][:2]
                    corner_pts[:, 0] = np.clip(corner_pts[:, 0], 0, results['img_shape'][1])
                    corner_pts[:, 1] = np.clip(corner_pts[:, 1], 0, results['img_shape'][0])
                    results[key] = corner_pts

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img','img_aug']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_wheels(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        img_shape = results['img_shape']
        scale_factor = np.array(results['scale_factor'].tolist()*2, dtype=np.float32).reshape(1, -1)
        for key in results.get('wheel_fields', []):
            if key == 'gt_wheels':
                bboxes = results[key] * scale_factor
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                results[key] = bboxes
    def _resize_ddd(self, results):
        for key in results.get('ddd_fields', []):
            # if key == 'gt_ddd_short_ys':
            #     bboxes = results[key] * results['scale_factor'][1]
            #     results[key] = bboxes
            if key in ['gt_ddds_l0', 'gt_ddds_l1', 'gt_ddds_l2']:
                bboxes = results[key] * results['scale_factor'][1]
                results[key] = bboxes
            elif key in ['gt_ddds_dx', 'gt_ddds_dw']:
                bboxes = results[key] * results['scale_factor'][0]
                results[key] = bboxes

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            # bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            # bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            bboxes[:, 0::2] = bboxes[:, 0::2]
            bboxes[:, 1::2] = bboxes[:, 1::2]
            results[key] = bboxes


    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['img_shape'][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    # (results['scale'][0]//2, results['scale'][1]//2),
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results[key],
                    results['scale'],
                    # (results['scale'][0]//2, results['scale'][1]//2),
                    interpolation='nearest',
                    backend=self.backend)
            results[key] = gt_seg
        # pass

    def _resize_car_mask(self, results):
        for key in results.get('car_mask_fields', []):
            if self.keep_ratio:
                car_mask = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                car_mask = mmcv.imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['car_mask'] = car_mask

    def _resize_depth(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('depth_fields', []):
            if self.keep_ratio:
                gt_depth = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_depth = mmcv.imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_depth'] = gt_depth

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            assert 'scale_factor' not in results, (
                'scale and scale_factor cannot be both set.')

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._resize_depth(results)
        self._resize_car_mask(results)
        self._resize_wheels(results)
        self._resize_ddd(results)
        self._resize_img_affine(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio})'
        return repr_str


@PIPELINES.register_module()
class RandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped


    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img','img_aug']):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip(results['flip_direction'])
            # flip optical flows
            for key in results.get('flow_fields', []):
                results[key] = mmcv.imflip(results[key], direction=results['flip_direction'])
                if results['flip_direction'] == 'horizontal':
                    results[key][..., 0] *= -1
                else:
                    results[key][..., 1] *= -1
            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
                if key == 'gt_lane_seg':
                    img_tmp = results[key].copy()
                    img_tmp[results[key]==1] = 4
                    img_tmp[results[key]==2] = 3
                    img_tmp[results[key]==3] = 2
                    img_tmp[results[key]==4] = 1
                    results[key] = img_tmp
                results[key] = results[key].copy()
                
            for key in results.get('depth_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])

            for key in results.get('car_mask_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
            
            for key in results.get('lane_fields', []):
                results[key] = results[key][::-1]
                results[key] = results[key].copy()
            
            # Todo
            for key in results.get('wheel_fields', []):
                if key == 'gt_wheels_exist':
                    label_tmp = results[key].copy()
                    label_tmp[..., 0] = results[key][..., 1]
                    label_tmp[..., 1] = results[key][..., 0]
                    label_tmp[..., 2] = results[key][..., 3]
                    label_tmp[..., 3] = results[key][..., 2]
                    results[key] = label_tmp
                else:
                    w = results['img_shape']
                    wheel_tmp = results[key].copy()
                    results[key][..., 0::4] = w - wheel_tmp[..., 2::4]
                    results[key][..., 2::4] = w - wheel_tmp[..., 0::4]

            for key in results.get('ddd_fields', []):
                if key == 'gt_ddds_head_direction':
                    label_tmp = results[key].copy()
                    for i in range(label_tmp.shape[0]):
                        if label_tmp[i, 0] == 1:
                            label_tmp[i, 0] = 0
                            label_tmp[i, 1] = 1
                        elif label_tmp[i, 1] == 1:
                            label_tmp[i, 1] = 0
                            label_tmp[i, 0] = 1
                    results[key] = label_tmp
                elif key == 'gt_ddds_dx':
                    label_tmp = results[key].copy()
                    results[key] = -label_tmp
                elif key == 'gt_ddds_center_2d':
                    w = results['img_shape'][1]
                    label_tmp = results[key].copy()
                    results[key][..., 0] = -label_tmp[..., 0]

                elif key == 'gt_ddd_rotations':
                    label_tmp = results[key].copy()
                    for i, item in enumerate(label_tmp):
                        if item < 0: 
                            results[key][i] = -np.pi - item
                        else:
                            results[key][i] = np.pi - item

            for key in results.get('seg_fields_resized', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
                results[key] = results[key].copy()
            
            if results.get('affine_fields', None):
                corner_pts = results['corner_pts'].copy()
                results['corner_pts'][:, 0] = results['img_shape'][1] - corner_pts[:, 0]
                
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


@PIPELINES.register_module()
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get('img_fields', ['img','img_aug']):
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=self.pad_val)
            results[key] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_flow(self, results):
        '''
        pad optical flow images 
        '''
        for key in results.get('flow_fields', []):
            if self.size is not None: 
                padded_flow = mmcv.impad(results[key], shape=self.size)
            elif self.size_divisor is not None:
                padded_flow = mmcv.impad_to_multiple(results[key], divisor=self.size_divisor)
            results[key] = padded_flow

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=self.pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2])

    def _pad_depth(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        for key in results.get('depth_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2])
    
    def _pad_car_mask(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        for key in results.get('car_mask_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2])

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_flow(results)
        self._pad_masks(results)
        self._pad_seg(results)
        self._pad_depth(results)
        self._pad_car_mask(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


# @PIPELINES.register_module()
# class NormalizeMinMax(object):
#     """Normalize the image.

#     Added key is "img_norm_cfg".

#     Args:
#         mean (sequence): Mean values of 3 channels.
#         std (sequence): Std values of 3 channels.
#         to_rgb (bool): Whether to convert the image from BGR to RGB,
#             default is true.
#     """

#     def __init__(self, to_rgb=True):
#         self.to_rgb = to_rgb

#     def __call__(self, results):
#         """Call function to normalize images.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Normalized results, 'img_norm_cfg' key is added into
#                 result dict.
#         """
#         for key in results.get('img_fields', ['img']):
#             results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
#                                             self.to_rgb)
#         results['img_norm_cfg'] = dict(
#             mean=self.mean, std=self.std, to_rgb=self.to_rgb)
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
#         return repr_str

@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img','img_aug']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class RandomCrop:
    """Random crop the image & bboxes & masks.
    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.
    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        crop_type (str, optional): one of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 bbox_clip_border=True,
                 center_crop=False,
                 thresh=1.):
        self.thresh = thresh
        self.center_crop = center_crop
        if crop_type not in [
                'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_2d': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_2d': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if 'crop_center' in results: 
                crop_center = results['crop_center']
                offset_h = crop_center[0] - crop_size[0] // 2 
                offset_w = crop_center[1] - crop_size[1] // 2
            else: 
                margin_h = max(img.shape[0] - crop_size[0], 0)
                margin_w = max(img.shape[1] - crop_size[1], 0)
                offset_h = np.random.randint(0, margin_h + 1)
                offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        if 'calib' in results:  
            # modify kitti calibration data
            calib = results['calib'] 
            P2 = calib['P2']
            # print('before: ', P2)
            P2[0, 2] -= crop_x1
            P2[1, 2] -= crop_y1
            # print('after: ', P2)
        return results

    def _get_crop_size(self, image_size):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.
        Args:
            image_size (tuple): (h, w).
        Returns:
            crop_size (tuple): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'absolute':
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == 'absolute_range':
            assert self.crop_size[0] <= self.crop_size[1]
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        elif self.crop_type == 'relative_range':
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        image_size = results['img'].shape[:2]

        if self.center_crop:
            results['crop_center'] = (image_size[0] // 2, image_size[1] // 2)
        if ('crop' in results and not results['crop']) or \
            ('crop' not in results and np.random.random() > self.thresh): 
            results['crop'] = False 
            return results
        results['crop'] = True
        crop_size = self._get_crop_size(image_size)

        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class RandomLaneCrop(object):
    """Random crop the image & bboxes & masks.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        allow_negative_crop (bool): Whether to allow a crop that does not
            contain any bbox area. Default to False.

    Note:
        - If the image is smaller than the crop size, return the original image
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self, crop_size, crop_ratio=0, crop_num=5):
        self.crop_size = crop_size # min_h, max_h, min_w, max_w
        self.crop_ratio = crop_ratio
        self.crop_num = crop_num

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        for i in range(self.crop_num):
            if np.random.rand() < self.crop_ratio:
                for key in results.get('img_fields', ['img','img_aug']):
                    if key == 'img_aug':
                        img = results[key]
                        start_y = np.random.randint(img.shape[0]//2, img.shape[0]-25)
                        end_y = min(np.random.randint(self.crop_size[0], self.crop_size[1])+start_y, img.shape[0]-1)

                        start_x = np.random.randint(80, img.shape[1]-80)
                        end_x = min(np.random.randint(self.crop_size[2], self.crop_size[3])+start_x, img.shape[1]-1)
                        img[start_y:end_y, start_x:end_x] = 0
                        results[key] = img

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

@PIPELINES.register_module()
class SegRescale(object):
    """Rescale semantic segmentation maps.

    Args:
        scale_factor (float): The scale factor of the final output.
        backend (str): Image rescale backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
    """

    def __init__(self, scale_factor=1, backend='cv2'):
        self.scale_factor = scale_factor
        self.backend = backend

    def __call__(self, results):
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """

        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key],
                    self.scale_factor,
                    interpolation='nearest',
                    backend=self.backend)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


@PIPELINES.register_module()
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,'\
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class Expand(object):
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
        prob (float): probability of applying this transformation
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 to_rgb=True,
                 ratio_range=(1, 4),
                 seg_ignore_label=None,
                 prob=0.5):
        self.to_rgb = to_rgb
        self.ratio_range = ratio_range
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob

    def __call__(self, results):
        """Call function to expand images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images, bounding boxes expanded
        """

        if random.uniform(0, 1) > self.prob:
            return results

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean,
                             dtype=img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img

        results['img'] = expand_img
        # expand bboxes
        for key in results.get('bbox_fields', []):
            results[key] = results[key] + np.tile(
                (left, top), 2).astype(results[key].dtype)
                
        # Todo wheel

        # expand masks
        for key in results.get('mask_fields', []):
            results[key] = results[key].expand(
                int(h * ratio), int(w * ratio), top, left)

        # expand segs
        for key in results.get('seg_fields', []):
            gt_seg = results[key]
            expand_gt_seg = np.full((int(h * ratio), int(w * ratio)),
                                    self.seg_ignore_label,
                                    dtype=gt_seg.dtype)
            expand_gt_seg[top:top + h, left:left + w] = gt_seg
            results[key] = expand_gt_seg
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, to_rgb={self.to_rgb}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label})'
        return repr_str


@PIPELINES.register_module()
class MinIoURandomCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).

    Note:
        The keys for bboxes, labels and masks should be paired. That is, \
        `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and \
        `gt_bboxes_ignore` to `gt_labels_ignore` and `gt_masks_ignore`.
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def __call__(self, results):
        """Call function to crop images and bounding boxes with minimum IoU
        constraint.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images and bounding boxes cropped, \
                'img_shape' key is updated.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert 'bbox_fields' in results
        boxes = [results[key] for key in results['bbox_fields']]
        boxes = np.concatenate(boxes, 0)
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        center = (boxes[:, :2] + boxes[:, 2:]) / 2
                        mask = ((center[:, 0] > patch[0]) *
                                (center[:, 1] > patch[1]) *
                                (center[:, 0] < patch[2]) *
                                (center[:, 1] < patch[3]))
                        return mask

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    for key in results.get('bbox_fields', []):
                        boxes = results[key].copy()
                        mask = is_center_of_bboxes_in_patch(boxes, patch)
                        boxes = boxes[mask]
                        boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                        boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                        boxes -= np.tile(patch[:2], 2)

                        results[key] = boxes
                        # labels
                        label_key = self.bbox2label.get(key)
                        if label_key in results:
                            results[label_key] = results[label_key][mask]

                        # mask fields
                        mask_key = self.bbox2mask.get(key)
                        if mask_key in results:
                            results[mask_key] = results[mask_key][
                                mask.nonzero()[0]].crop(patch)
                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                results['img'] = img
                results['img_shape'] = img.shape

                # seg fields
                for key in results.get('seg_fields', []):
                    results[key] = results[key][patch[1]:patch[3],
                                                patch[0]:patch[2]]
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_ious={self.min_ious}, '
        repr_str += f'min_crop_size={self.min_crop_size})'
        return repr_str


@PIPELINES.register_module()
class Corrupt(object):
    """Corruption augmentation.

    Corruption transforms implemented based on
    `imagecorruptions <https://github.com/bethgelab/imagecorruptions>`_.

    Args:
        corruption (str): Corruption name.
        severity (int, optional): The severity of corruption. Default: 1.
    """

    def __init__(self, corruption, severity=1):
        self.corruption = corruption
        self.severity = severity

    def __call__(self, results):
        """Call function to corrupt image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images corrupted.
        """

        if corrupt is None:
            raise RuntimeError('imagecorruptions is not installed')
        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        results['img'] = corrupt(
            results['img'].astype(np.uint8),
            corruption_name=self.corruption,
            severity=self.severity)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(corruption={self.corruption}, '
        repr_str += f'severity={self.severity})'
        return repr_str


@PIPELINES.register_module()
class Albu(object):
    """Albumentation augmentation.

    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.

    An example of ``transforms`` is as followed:

    .. code-block::

        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]

    Args:
        transforms (list[dict]): A list of albu transformations
        bbox_params (dict): Bbox_params for albumentation `Compose`
        keymap (dict): Contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): Whether to skip the image if no ann left
            after aug
    """

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 additional_targets=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params,
                           additional_targets=additional_targets)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes_2d': 'bboxes'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper. Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        # TODO: add bbox_fields
        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        # TODO: Support mask structure in albu
        if 'masks' in results:
            if isinstance(results['masks'], PolygonMasks):
                raise NotImplementedError(
                    'Albu only supports BitMap masks now')
            ori_masks = results['masks']
            results['masks'] = results['masks'].masks

        results = self.aug(**results)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)

            # filter label_fields
            if self.filter_lost_elements:

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    results['masks'] = np.array(
                        [results['masks'][i] for i in results['idx_mapper']])
                    results['masks'] = ori_masks.__class__(
                        results['masks'], results['image'].shape[0],
                        results['image'].shape[1])

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])
            results['gt_labels'] = results['gt_labels'].astype(np.int64)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@PIPELINES.register_module()
class RandomCenterCropPad(object):
    """Random center crop and random around padding for CornerNet.

    This operation generates randomly cropped image from the original image and
    pads it simultaneously. Different from :class:`RandomCrop`, the output
    shape may not equal to ``crop_size`` strictly. We choose a random value
    from ``ratios`` and the output shape could be larger or smaller than
    ``crop_size``. The padding operation is also different from :class:`Pad`,
    here we use around padding instead of right-bottom padding.

    The relation between output image (padding image) and original image:

    .. code:: text

                        output image

               +----------------------------+
               |          padded area       |
        +------|----------------------------|----------+
        |      |         cropped area       |          |
        |      |         +---------------+  |          |
        |      |         |    .   center |  |          | original image
        |      |         |        range  |  |          |
        |      |         +---------------+  |          |
        +------|----------------------------|----------+
               |          padded area       |
               +----------------------------+

    There are 5 main areas in the figure:

    - output image: output image of this operation, also called padding
      image in following instruction.
    - original image: input image of this operation.
    - padded area: non-intersect area of output image and original image.
    - cropped area: the overlap of output image and original image.
    - center range: a smaller area where random center chosen from.
      center range is computed by ``border`` and original image's shape
      to avoid our random center is too close to original image's border.

    Also this operation act differently in train and test mode, the summary
    pipeline is listed below.

    Train pipeline:

    1. Choose a ``random_ratio`` from ``ratios``, the shape of padding image
       will be ``random_ratio * crop_size``.
    2. Choose a ``random_center`` in center range.
    3. Generate padding image with center matches the ``random_center``.
    4. Initialize the padding image with pixel value equals to ``mean``.
    5. Copy the cropped area to padding image.
    6. Refine annotations.

    Test pipeline:

    1. Compute output shape according to ``test_pad_mode``.
    2. Generate padding image with center matches the original image
       center.
    3. Initialize the padding image with pixel value equals to ``mean``.
    4. Copy the ``cropped area`` to padding image.

    Args:
        crop_size (tuple | None): expected size after crop, final size will
            computed according to ratio. Requires (h, w) in train mode, and
            None in test mode.
        ratios (tuple): random select a ratio from tuple and crop image to
            (crop_size[0] * ratio) * (crop_size[1] * ratio).
            Only available in train mode.
        border (int): max distance from center select area to image border.
            Only available in train mode.
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB.
        test_mode (bool): whether involve random variables in transform.
            In train mode, crop_size is fixed, center coords and ratio is
            random selected from predefined lists. In test mode, crop_size
            is image's original shape, center coords and ratio is fixed.
        test_pad_mode (tuple): padding method and padding shape value, only
            available in test mode. Default is using 'logical_or' with
            127 as padding shape value.

            - 'logical_or': final_shape = input_shape | padding_shape_value
            - 'size_divisor': final_shape = int(
              ceil(input_shape / padding_shape_value) * padding_shape_value)
    """

    def __init__(self,
                 crop_size=None,
                 ratios=(0.9, 1.0, 1.1),
                 border=128,
                 mean=None,
                 std=None,
                 to_rgb=None,
                 test_mode=False,
                 test_pad_mode=('logical_or', 127)):
        if test_mode:
            assert crop_size is None, 'crop_size must be None in test mode'
            assert ratios is None, 'ratios must be None in test mode'
            assert border is None, 'border must be None in test mode'
            assert isinstance(test_pad_mode, (list, tuple))
            assert test_pad_mode[0] in ['logical_or', 'size_divisor']
        else:
            assert isinstance(crop_size, (list, tuple))
            assert crop_size[0] > 0 and crop_size[1] > 0, (
                'crop_size must > 0 in train mode')
            assert isinstance(ratios, (list, tuple))
            assert test_pad_mode is None, (
                'test_pad_mode must be None in train mode')

        self.crop_size = crop_size
        self.ratios = ratios
        self.border = border
        # We do not set default value to mean, std and to_rgb because these
        # hyper-parameters are easy to forget but could affect the performance.
        # Please use the same setting as Normalize for performance assurance.
        assert mean is not None and std is not None and to_rgb is not None
        self.to_rgb = to_rgb
        self.input_mean = mean
        self.input_std = std
        if to_rgb:
            self.mean = mean[::-1]
            self.std = std[::-1]
        else:
            self.mean = mean
            self.std = std
        self.test_mode = test_mode
        self.test_pad_mode = test_pad_mode

    def _get_border(self, border, size):
        """Get final border for the target size.

        This function generates a ``final_border`` according to image's shape.
        The area between ``final_border`` and ``size - final_border`` is the
        ``center range``. We randomly choose center from the ``center range``
        to avoid our random center is too close to original image's border.
        Also ``center range`` should be larger than 0.

        Args:
            border (int): The initial border, default is 128.
            size (int): The width or height of original image.
        Returns:
            int: The final border.
        """
        k = 2 * border / size
        i = pow(2, np.ceil(np.log2(np.ceil(k))) + (k == int(k)))
        return border // i

    def _filter_boxes(self, patch, boxes):
        """Check whether the center of each box is in the patch.

        Args:
            patch (list[int]): The cropped area, [left, top, right, bottom].
            boxes (numpy array, (N x 4)): Ground truth boxes.

        Returns:
            mask (numpy array, (N,)): Each box is inside or outside the patch.
        """
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask = (center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) * (
            center[:, 0] < patch[2]) * (
                center[:, 1] < patch[3])
        return mask

    def _crop_image_and_paste(self, image, center, size):
        """Crop image with a given center and size, then paste the cropped
        image to a blank image with two centers align.

        This function is equivalent to generating a blank image with ``size``
        as its shape. Then cover it on the original image with two centers (
        the center of blank image and the random center of original image)
        aligned. The overlap area is paste from the original image and the
        outside area is filled with ``mean pixel``.

        Args:
            image (np array, H x W x C): Original image.
            center (list[int]): Target crop center coord.
            size (list[int]): Target crop size. [target_h, target_w]

        Returns:
            cropped_img (np array, target_h x target_w x C): Cropped image.
            border (np array, 4): The distance of four border of
                ``cropped_img`` to the original image area, [top, bottom,
                left, right]
            patch (list[int]): The cropped area, [left, top, right, bottom].
        """
        center_y, center_x = center
        target_h, target_w = size
        img_h, img_w, img_c = image.shape

        x0 = max(0, center_x - target_w // 2)
        x1 = min(center_x + target_w // 2, img_w)
        y0 = max(0, center_y - target_h // 2)
        y1 = min(center_y + target_h // 2, img_h)
        patch = np.array((int(x0), int(y0), int(x1), int(y1)))

        left, right = center_x - x0, x1 - center_x
        top, bottom = center_y - y0, y1 - center_y

        cropped_center_y, cropped_center_x = target_h // 2, target_w // 2
        cropped_img = np.zeros((target_h, target_w, img_c), dtype=image.dtype)
        for i in range(img_c):
            cropped_img[:, :, i] += self.mean[i]
        y_slice = slice(cropped_center_y - top, cropped_center_y + bottom)
        x_slice = slice(cropped_center_x - left, cropped_center_x + right)
        cropped_img[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

        border = np.array([
            cropped_center_y - top, cropped_center_y + bottom,
            cropped_center_x - left, cropped_center_x + right
        ],
                          dtype=np.float32)

        return cropped_img, border, patch

    def _train_aug(self, results):
        """Random crop and around padding the original image.

        Args:
            results (dict): Image infomations in the augment pipeline.

        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        h, w, c = img.shape
        boxes = results['gt_bboxes']
        while True:
            scale = random.choice(self.ratios)
            new_h = int(self.crop_size[0] * scale)
            new_w = int(self.crop_size[1] * scale)
            h_border = self._get_border(self.border, h)
            w_border = self._get_border(self.border, w)

            for i in range(50):
                center_x = random.randint(low=w_border, high=w - w_border)
                center_y = random.randint(low=h_border, high=h - h_border)

                cropped_img, border, patch = self._crop_image_and_paste(
                    img, [center_y, center_x], [new_h, new_w])

                mask = self._filter_boxes(patch, boxes)
                # if image do not have valid bbox, any crop patch is valid.
                if not mask.any() and len(boxes) > 0:
                    continue

                results['img'] = cropped_img
                results['img_shape'] = cropped_img.shape
                results['pad_shape'] = cropped_img.shape

                x0, y0, x1, y1 = patch

                left_w, top_h = center_x - x0, center_y - y0
                cropped_center_x, cropped_center_y = new_w // 2, new_h // 2

                # crop bboxes accordingly and clip to the image boundary
                for key in results.get('bbox_fields', []):
                    mask = self._filter_boxes(patch, results[key])
                    bboxes = results[key][mask]
                    bboxes[:, 0:4:2] += cropped_center_x - left_w - x0
                    bboxes[:, 1:4:2] += cropped_center_y - top_h - y0
                    bboxes[:, 0:4:2] = np.clip(bboxes[:, 0:4:2], 0, new_w)
                    bboxes[:, 1:4:2] = np.clip(bboxes[:, 1:4:2], 0, new_h)
                    keep = (bboxes[:, 2] > bboxes[:, 0]) & (
                        bboxes[:, 3] > bboxes[:, 1])
                    bboxes = bboxes[keep]
                    results[key] = bboxes
                    if key in ['gt_bboxes']:
                        if 'gt_labels' in results:
                            labels = results['gt_labels'][mask]
                            labels = labels[keep]
                            results['gt_labels'] = labels
                        if 'gt_masks' in results:
                            raise NotImplementedError(
                                'RandomCenterCropPad only supports bbox.')

                # crop semantic seg
                for key in results.get('seg_fields', []):
                    raise NotImplementedError(
                        'RandomCenterCropPad only supports bbox.')
                return results

    def _test_aug(self, results):
        """Around padding the original image without cropping.

        The padding mode and value are from ``test_pad_mode``.

        Args:
            results (dict): Image infomations in the augment pipeline.

        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        h, w, c = img.shape
        results['img_shape'] = img.shape
        if self.test_pad_mode[0] in ['logical_or']:
            target_h = h | self.test_pad_mode[1]
            target_w = w | self.test_pad_mode[1]
        elif self.test_pad_mode[0] in ['size_divisor']:
            divisor = self.test_pad_mode[1]
            target_h = int(np.ceil(h / divisor)) * divisor
            target_w = int(np.ceil(w / divisor)) * divisor
        else:
            raise NotImplementedError(
                'RandomCenterCropPad only support two testing pad mode:'
                'logical-or and size_divisor.')

        cropped_img, border, _ = self._crop_image_and_paste(
            img, [h // 2, w // 2], [target_h, target_w])
        results['img'] = cropped_img
        results['pad_shape'] = cropped_img.shape
        results['border'] = border
        return results

    def __call__(self, results):
        img = results['img']
        assert img.dtype == np.float32, (
            'RandomCenterCropPad needs the input image of dtype np.float32,'
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline')
        h, w, c = img.shape
        assert c == len(self.mean)
        if self.test_mode:
            return self._test_aug(results)
        else:
            return self._train_aug(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'ratios={self.ratios}, '
        repr_str += f'border={self.border}, '
        repr_str += f'mean={self.input_mean}, '
        repr_str += f'std={self.input_std}, '
        repr_str += f'to_rgb={self.to_rgb}, '
        repr_str += f'test_mode={self.test_mode}, '
        repr_str += f'test_pad_mode={self.test_pad_mode})'
        return repr_str

@PIPELINES.register_module()
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.1, sh=0.4, r1= 0.2, mean=[0.406, 0.456,0.485]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def earse(self,img,bboxes_2d):
        # img: h,w,c
        for i in range(len(bboxes_2d)):
            if np.random.rand() < 0.6:
                bbox = bboxes_2d[i]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area <= 400:
                    continue
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)
                h = int(round(np.sqrt(target_area * aspect_ratio)))
                w = int(round(np.sqrt(target_area / aspect_ratio)))
                if w < int(bbox[2]-bbox[0]) and h < int(bbox[3]-bbox[1]):
                    x1 = random.randint(int(bbox[0]), int(bbox[2] - w))
                    y1 = random.randint(int(bbox[1]), int(bbox[3] - h))
                    if img.shape[2] == 3:
                        img[y1:y1 + h, x1:x1 + w, ] = np.random.uniform(0, 255, (h, w, 3))
                    else:
                        img[y1:y1 + h, x1:x1 + w, 0] = np.random.uniform(0, 255, (h, w))
        return img

    def __call__(self, results):
        erase = True if np.random.rand() < self.probability else False
        if erase:
            for key in results.get('img_fields', ['img', 'img_aug']):
                    bboxes_2d = results['gt_bboxes_2d']
                    if bboxes_2d.shape[0] == 0:
                        continue
                    else:
                        results[key] = self.earse(results[key],bboxes_2d)
                # if key == 'img_aug':
                #     bboxes_2d = results['gt_bboxes_2d']
                #     if bboxes_2d.shape[0] == 0:
                #         continue
                #     else:
                #         results[key] = self.earse(results[key],bboxes_2d)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(erase_ratio={self.probability})'
        return repr_str


@PIPELINES.register_module()
class RandomCropResize(object):

    def __init__(self, probability=1, threshold=0.6):

        self.probability = probability
        self.threshold = threshold

    def crop_resize(self ,img, ori_h, ori_w, crop_h, crop_w, minx, miny, dims=3):
        if dims == 3:
            img = img[:,minx:minx+crop_w+1,:]
            img = img[miny:miny+crop_h+1,:,:]
        elif dims == 1:
            img = img[:,minx:minx+crop_w+1]
            img = img[miny:miny+crop_h+1,:]
        else:
            print("dimentions of object is not 1 or 3")
        img = cv2.resize(img ,(ori_w ,ori_h))
        return img

    def bbox_transform(self, bbox, crop_h, crop_w, minx, miny, scale_factor):
        bbox[:,0] = bbox[:,0].clip(minx,minx+crop_w)
        bbox[:,2] = bbox[:,2].clip(minx,minx+crop_w)
        bbox[:,1] = bbox[:,1].clip(miny,miny+crop_h)
        bbox[:,3] = bbox[:,3].clip(miny,miny+crop_h)
        bbox[:,0] = bbox[:,0] - minx
        bbox[:,2] = bbox[:,2] - minx
        bbox[:,1] = bbox[:,1] - miny
        bbox[:,3] = bbox[:,3] - miny
        bbox = bbox/scale_factor
        return bbox

    def filter_anns(self, bbox, label, reid, crop_h, crop_w, minx, miny):

        jug_x_1 = (bbox[:,2]>minx)
        bbox = bbox[jug_x_1]
        label = label[jug_x_1]
        reid = reid[jug_x_1]
        jug_x_2 = (bbox[:,0]<minx+crop_w)
        bbox = bbox[jug_x_2]
        label = label[jug_x_2]
        reid = reid[jug_x_2]
        jug_y_1 = (bbox[:,3]>miny)
        bbox = bbox[jug_y_1]
        label = label[jug_y_1]
        reid = reid[jug_y_1]
        jug_y_2 = (bbox[:,1]<miny+crop_h)
        bbox = bbox[jug_y_2]
        label = label[jug_y_2]
        reid = reid[jug_y_2]

        return bbox,label,reid
    
    def __call__(self, results):
        crop = True if np.random.rand() < self.probability else False
        if crop:
            scale = random.uniform(0.5,1)
            min_scale_x = random.uniform(0, 1-scale)
            min_scale_y = random.uniform(0, 1-scale)
            for key in results.get('img_fields', ['img', 'img_aug']):
                ori_h, ori_w, _ = results[key].shape
                minx = int(ori_w * min_scale_x)
                miny = int(ori_h * min_scale_y)
                crop_h = int(ori_h * scale) if miny + int(ori_h * scale) <= ori_h else ori_h - miny
                crop_w = int(ori_w * scale) if minx + int(ori_w * scale) <= ori_w else ori_w - minx
                results[key] = self.crop_resize(img=results[key], ori_h=ori_h, ori_w=ori_w,
                                                crop_h=crop_h, crop_w=crop_w, minx=minx, miny=miny,dims=3)
                for sub_key in results.get('bbox_fields',[]):
                    if sub_key is 'gt_bboxes_2d':
                        bboxes = results[sub_key]
                        labels = results["gt_labels"]
                        
                        re_ids = results["gt_reids"]
                        bboxes,results["gt_labels"],results["gt_reids"] = self.filter_anns(bboxes, labels, re_ids,
                                                                                           crop_h, crop_w, minx, miny)

            for key in results.get('seg_fields' ,[]):
                ori_h, ori_w = results[key].shape
                minx = int(ori_w * min_scale_x)
                miny = int(ori_h * min_scale_y)
                crop_h = int(ori_h * scale) if miny + int(ori_h * scale) <= ori_h else ori_h - miny
                crop_w = int(ori_w * scale) if minx + int(ori_w * scale) <= ori_w else ori_w - minx
                results[key] = self.crop_resize(results[key], ori_h=ori_h, ori_w=ori_w,
                                                crop_h=crop_h, crop_w=crop_w, minx=minx, miny=miny,dims=1)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(erase_ratio={self.probability})'
        return repr_str


###### ultra-fast-lane ######
@PIPELINES.register_module()
class LaneRandomRotate(object):
    def __init__(self, angle):
        self.angle = angle

    def imrotate(self, img,
             angle,
             center=None,
             scale=1.0,
             border_value=0,
             interpolation='bilinear',
             auto_bound=False):
        """Rotate an image.
        Args:
            img (ndarray): Image to be rotated.
            angle (float): Rotation angle in degrees, positive values mean
                clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the rotation in
                the source image. If not specified, the center of the image will be
                used.
            scale (float): Isotropic scale factor.
            border_value (int): Border value.
            interpolation (str): Same as :func:`resize`.
            auto_bound (bool): Whether to adjust the image size to cover the whole
                rotated image.
        Returns:
            ndarray: The rotated image.
        """
        import cv2
        cv2_interp_codes = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }

        if center is not None and auto_bound:
            raise ValueError('`auto_bound` conflicts with `center`')
        h, w = img.shape[:2]
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        assert isinstance(center, tuple)

        matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        if auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))
        rotated = cv2.warpAffine(
            img,
            matrix, (w, h),
            flags=cv2_interp_codes[interpolation],
            borderValue=border_value)
        return rotated

    def __call__(self, results):
        angle = random.randint(0, self.angle * 2) - self.angle

        for key in results.get('img_fields', ['img_aug']):
            if key == 'img_aug':
                image = self.imrotate(results[key], angle)
                results[key] = image

        for key in results.get('seg_fields', ['gt_lane_seg']):
            if key == 'gt_lane_seg':
                label = self.imrotate(results[key], angle, interpolation='nearest')
                results[key] = label

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(angle={self.angle})'
        return repr_str

        
@PIPELINES.register_module()
class RandomLROffsetLABEL(object):
    def __init__(self,max_offset):
        self.max_offset = max_offset
        self.counter = 0
    def __call__(self, results):
        offset = np.random.randint(-self.max_offset,self.max_offset)
        h, w, _ = results['img'].shape
        for key in results.get('img_fields', ['img_aug']):
            if key == 'img_aug':
                img = results[key].copy()
                if offset > 0:
                    img[:,offset:,:] = img[:,0:w-offset,:]
                    img[:,:offset,:] = 0
                if offset < 0:
                    real_offset = -offset
                    img[:,0:w-real_offset,:] = img[:,real_offset:,:]
                    img[:,w-real_offset:,:] = 0

                results[key] = img

        for key in results.get('seg_fields', ['gt_lane_seg']):
            if key == 'gt_lane_seg':
                label = results[key].copy()
                if offset > 0:
                    label[:,offset:] = label[:,0:w-offset]
                    label[:,:offset] = 0
                if offset < 0:
                    real_offset = -offset
                    label[:,0:w-real_offset] = label[:,real_offset:]
                    label[:,w-real_offset:] = 0

                results[key] = label
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_offset={self.max_offset})'
        return repr_str


@PIPELINES.register_module()
class RandomUDoffsetLABEL(object):
    def __init__(self,max_offset):
        self.max_offset = max_offset
        self.counter = 0
    def __call__(self, results):
        offset = np.random.randint(-self.max_offset,self.max_offset)
        h, w, _ = results['img'].shape
        for key in results.get('img_fields', ['img_aug']):
            if key == 'img_aug':
                img = results[key].copy()
                if offset > 0:
                    img[offset:,:,:] = img[0:h-offset,:,:]
                    img[:offset,:,:] = 0
                if offset < 0:
                    real_offset = -offset
                    img[0:h-real_offset,:,:] = img[real_offset:,:,:]
                    img[h-real_offset:,:,:] = 0

                results[key] = img

        for key in results.get('seg_fields', ['gt_lane_seg']):
            if key == 'gt_lane_seg':
                label = results[key].copy()
                if offset > 0:
                    label[offset:,:] = label[0:h-offset,:]
                    label[:offset,:] = 0
                if offset < 0:
                    real_offset = -offset
                    label[0:h-real_offset,:] = label[real_offset:,:]
                    label[h-real_offset:,:] = 0

                results[key] = label
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_offset={self.max_offset})'
        return repr_str


### ultra-fast-lane-detection
@PIPELINES.register_module()
class FormatUltraFastLane(object):
    def __init__(self, row_anchor=None):
        self.row_anchor = row_anchor

    def lane_grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)
        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def find_start_pos(self, row_sample,start_line):
        # row_sample = row_sample.sort()
        # for i,r in enumerate(row_sample):
        #     if r >= start_line:
        #         return i
        l,r = 0,len(row_sample)-1
        while True:
            mid = int((l+r)/2)
            if r - l == 1:
                return r
            if row_sample[mid] < start_line:
                l = mid
            if row_sample[mid] > start_line:
                r = mid
            if row_sample[mid] == start_line:
                return mid

    def lane_get_index(self, label):
        
        h, w = label.shape
    
        if h != 512:
            scale_f = lambda x : int((x * 1.0/512) * h)
            sample_tmp = list(map(scale_f, self.row_anchor))
        else:
            sample_tmp = self.row_anchor
            
        all_idx = np.zeros((4,len(sample_tmp),2))
        for i,r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, 5):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos
        all_idx_cp = all_idx.copy()
        for i in range(4):
            if np.all(all_idx_cp[i,:,1] == -1):
                continue

            valid = all_idx_cp[i,:,1] != -1
            valid_idx = all_idx_cp[i,valid,:]
            if valid_idx[-1,0] == all_idx_cp[0,-1,0]:
                continue
            if len(valid_idx) < 6:
                continue

            valid_idx_half = valid_idx[len(valid_idx) // 2:,:]
            p = np.polyfit(valid_idx_half[:,0], valid_idx_half[:,1],deg = 1)
            start_line = valid_idx_half[-1,0]
            pos = self.find_start_pos(all_idx_cp[i,:,0],start_line) + 1
            
            fitted = np.polyval(p,all_idx_cp[i,pos:,0])
            fitted = np.array([-1  if y < 0 or y > w-1 else y for y in fitted])

            assert np.all(all_idx_cp[i,pos:,1] == -1)
            all_idx_cp[i,pos:,1] = fitted
        
        return all_idx_cp

    def __call__(self, results):
        lane_pts = self.lane_get_index(results['gt_lane_seg'].copy())
        results['gt_lane_pts'] = lane_pts
        h, w = results['gt_lane_seg'].shape
        cls_label = self.lane_grid_pts(lane_pts, 200, w)
        results['gt_cls_label'] = cls_label
        results['lane_fields'].append('gt_cls_label')
        results['lane_fields'].append('gt_lane_pts')
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'
