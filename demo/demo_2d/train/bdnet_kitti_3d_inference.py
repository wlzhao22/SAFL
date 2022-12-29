from mmdet.apis import inference_detector
from mmdet.models import build_detector
from mmcv.runner import load_state_dict
from collections import OrderedDict
import scipy
import mmcv
import os
import torch
import numpy as np
import cv2
import time
import random
from mmdet.models.utils.efficientdet_utils import _topk, tranpose_and_gather_feat, nms_torch, plot_one_box
from mmdet.models.utils.kitti_vis import *
from mmdet.models.utils.kitti_utils import *
import json

def anchor_based_post_process(det_list, scale):
    result_list = []
    thr_hd = 0.5
    thr_box = 0.6
    
    for det in det_list:
        bbox, label, score = det['bbox'], det['label'], det['score']

        if score <= thr_box:
            continue

        x1, y1, x2, y2 = bbox
        x1 /= scale[0]
        y1 /= scale[1]
        x2 /= scale[0]
        y2 /= scale[1]

        dw, l0, l1, l2, dx, res_x, res_y, h, w, l, head_score, rotation_res, rotation_center = det['dw'], det['l0'], det['l1'], det['l2'], det['dx'], det['res_x'],  det['res_y'], det['h'], det['w'], det['l'], det['head_score'], det['rotation_res'], det['rotation_center'] 

        dw /= scale[0]
        l0 /= scale[1]
        l1 /= scale[1]
        l2 /= scale[1]
        dx /= scale[0]
        res_x /= scale[0]
        res_y /= scale[1]

        rotation = rotation_center - np.arctan(rotation_res)
        if rotation > np.pi:
            rotation -= 2*np.pi
        elif rotation < -np.pi:
            rotation += 2*np.pi

        # left front
        if head_score[0] >= head_score[1] and head_score[0] > thr_hd and head_score[2] >= head_score[3] and head_score[2] > thr_hd:
            head_label = [1, 0, 1, 0]

        # right front
        elif head_score[1] >= head_score[0] and head_score[1] > thr_hd and head_score[2] >= head_score[3] and head_score[2] > thr_hd:
            head_label = [0, 1, 1, 0]

        # left rear
        elif head_score[0] >= head_score[1] and head_score[0] > thr_hd and head_score[3] >= head_score[2] and head_score[3] > thr_hd:
            head_label = [1, 0, 0, 1]

        # right rear
        elif head_score[1] >= head_score[0] and head_score[1] > thr_hd and head_score[3] >= head_score[2] and head_score[3] > thr_hd:
            head_label = [0, 1, 0, 1]

        # front
        elif head_score[0] <= thr_hd and head_score[1] <= thr_hd and head_score[2] >= head_score[3] and head_score[2] > thr_hd:
            head_label = [0, 0, 1, 0]

        # rear
        elif head_score[0] <= thr_hd and head_score[1] <= thr_hd and head_score[3] >= head_score[2] and head_score[3] > thr_hd:
            head_label = [0, 0, 0, 1]
        
        # else
        else:
            continue
            
        result_list.append(np.array([x1, y1, x2, y2, score, label, dw, l0, l1, l2, dx, res_x, res_y, h, w, l,] + head_label + [rotation]))

    return np.array(result_list)


def compute_box_3d_2(ry, l, w, h):
    R = roty(ry)
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, 0, -l / 2, 0, ]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, 0, -w / 2, 0, w / 2, ]

    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))

    return corners_3d.T

def display(bbox, img, calib, top_view):
    if bbox.shape[0] == 0:
        return img, top_view

    color_0 = (255, 255, 255)  # tail
    color_1 = (101, 67, 254)  # tail
    color_2 = (0, 208, 244)
    color_3 = (255, 255, 255)  # front
    line_thickness = 1
    P = calib.P
    pinv_P = np.linalg.pinv(P)

    for j in range(len(bbox)):
        xmin, ymin, xmax, ymax = bbox[j,:4].astype(np.int)
        score = float(bbox[j,4])
        label = obj_list[int(bbox[j,5])]
        dw = int(bbox[j, 6])
        l0 = int(bbox[j, 7])
        l1 = int(bbox[j, 8])
        l2 = int(bbox[j, 9])
        l3 = l2
        dx = int(bbox[j, 10])
        res_x = int(bbox[j, 11])
        res_y = int(bbox[j, 12])
        h, w, l = bbox[j, 13:16]
        clazz = bbox[j, 16:20]
        rotation = bbox[j, 20]
        bbox_3d = compute_box_3d_2(rotation, l, w, h)

        c_x = (xmin + xmax) // 2 - res_x
        c_y = (ymin + ymax) // 2 - res_y
        cv2.circle(img, (c_x, c_y), 2, color_2, 2)

        if clazz[0] == 1 and clazz[1] == 0 and clazz[2] == 1 and clazz[3] == 0:  # left and leftFront
            # 0------1------2
            # |      |      |
            # |      |      |
            # |      |      3
            # |      |     /
            # 4------5/

            x_center = (xmin + xmax) // 2 - dx
            p0 = (x_center - dw, ymin)
            p1 = (x_center + dw, ymin)
            p2 = (xmax, ymin + l0)
            p3 = (xmax, ymin + l0 + l1)
            p4 = (x_center - dw, ymin + l2)
            p5 = (x_center + dw, ymin + l3)

            p2_c, p2_c_z = np.array([(p1[0] + p5[0]) / 2, (p1[1] + p5[1]) / 2, 1]), h * P[1, 1] / l3
            p3_c, p3_c_z = np.array([(p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2, 1]), h * P[1, 1] / l1

            c_z = (p2_c_z + p3_c_z) / 2
            c_x = (xmin + xmax) / 2 - res_x
            c_y = (ymin + ymax) / 2 - res_y

            p_c = np.array([c_x, c_y, c_z])
            p_c_3d = calib.project_image_to_rect_single(p_c)

            bbox_3d = (bbox_3d.T + p_c_3d.reshape(3, 1)).T

            cv2.line(img, p0, p1, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p5, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p5, p4, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p0, p4, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p2, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p2, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p3, p5, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)

        elif clazz[0] == 0 and clazz[1] == 0 and clazz[2] == 1 and clazz[3] == 0:  # front
            # 0------1
            # |      |
            # |      |
            # |      |
            # |      |
            # 2------3
            p0 = (xmin, ymin)
            p1 = (xmax, ymin)
            p2 = (xmin, ymax)
            p3 = (xmax, ymax)

            p1_c, p1_c_z = np.array([(p0[0] + p1[0]) / 2, (p0[1] + p2[1]) / 2, 1]), h * P[1, 1] / (ymax - ymin)
            p2_c_z = w * P[0, 0] / (xmax - xmin)

            p_c = np.array([(p0[0] + p1[0]) / 2, (p0[1] + p2[1]) / 2, p1_c_z])
            p1_3d = calib.project_image_to_rect_single(p_c)

            ct_3d = p1_3d[:3] - bbox_3d[8]
            bbox_3d = (bbox_3d.T + ct_3d.reshape(3, 1)).T

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_3, line_thickness)

        elif clazz[0] == 0 and clazz[1] == 1 and clazz[2] == 1 and clazz[3] == 0:  # right and rightFront
            # 0------1------2
            # |      |      |
            # |      |      |
            # 3      |      |
            #    \   |      |
            #      \ 4------5
            x_center = (xmin + xmax) // 2 - dx
            p0 = (xmin, ymin + l0)
            p1 = (x_center - dw, ymin)
            p2 = (x_center + dw, ymin)
            p3 = (xmin, ymin + l0 + l1)
            p4 = (x_center - dw, ymin + l2)
            p5 = (x_center + dw, ymin + l3)

            p1_c, p1_c_z = np.array([(p1[0] + p4[0]) / 2, (p1[1] + p4[1]) / 2, 1]), h * P[1, 1] / l2
            p2_c, p2_c_z = np.array([(p2[0] + p5[0]) / 2, (p2[1] + p5[1]) / 2, 1]), h * P[1, 1] / l3
            p3_c, p3_c_z = np.array([(p0[0] + p3[0]) / 2, (p0[1] + p3[1]) / 2, 1]), h * P[1, 1] / l1

            c_z = (p1_c_z + p3_c_z) / 2
            c_x = (xmin + xmax) / 2 - res_x
            c_y = (ymin + ymax) / 2 - res_y

            p_c = np.array([c_x, c_y, c_z])
            p_c_3d = calib.project_image_to_rect_single(p_c)

            bbox_3d = (bbox_3d.T + p_c_3d.reshape(3, 1)).T

            cv2.line(img, p1, p2, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p2, p5, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p5, p4, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p4, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p0, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p0, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p4, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)

        elif clazz[0] == 1 and clazz[1] == 0 and clazz[2] == 0 and clazz[3] == 1:  # leftRear
            # 0------1------2
            # |      |      |
            # |      |      |
            # 3      |      |
            #    \   |      |
            #      \ 4------5
            x_center = (xmin + xmax) // 2 - dx
            p0 = (xmin, ymin + l0)
            p1 = (x_center - dw, ymin)
            p2 = (x_center + dw, ymin)
            p3 = (xmin, ymin + l1 + l0)
            p4 = (x_center - dw, ymin + l2)
            p5 = (x_center + dw, ymin + l3)

            p1_c, p1_c_z = np.array([(p1[0] + p4[0]) / 2, (p1[1] + p4[1]) / 2, 1]), h * P[1, 1] / l2
            p2_c, p2_c_z = np.array([(p2[0] + p5[0]) / 2, (p2[1] + p5[1]) / 2, 1]), h * P[1, 1] / l3
            p3_c, p3_c_z = np.array([(p0[0] + p3[0]) / 2, (p0[1] + p3[1]) / 2, 1]), h * P[1, 1] / l1

            c_z = (p1_c_z + p3_c_z) / 2
            c_x = (xmin + xmax) / 2 - res_x
            c_y = (ymin + ymax) / 2 - res_y

            p_c = np.array([c_x, c_y, c_z])
            p_c_3d = calib.project_image_to_rect_single(p_c)

            bbox_3d = (bbox_3d.T + p_c_3d.reshape(3, 1)).T

            cv2.line(img, p1, p2, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p2, p5, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p5, p4, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p4, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p0, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p0, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p4, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)

        elif clazz[0] == 0 and clazz[1] == 0 and clazz[2] == 0 and clazz[3] == 1:  # rear
            # 0------1
            # |      |
            # |      |
            # |      |
            # |      |
            # 2------3

            p0 = (xmin, ymin)
            p1 = (xmax, ymin)
            p2 = (xmin, ymax)
            p3 = (xmax, ymax)

            p1_c, p1_c_z = np.array([(p0[0] + p1[0]) / 2, (p0[1] + p2[1]) / 2, 1]), h * P[1, 1] / (ymax - ymin)
            # p1_3d = np.dot(pinv_P, p1_c.T * p1_c_z)
            # p1_3d = np.dot(pinv_P, p1_c.T * center_3d[-1])
            p2_c_z = w * P[0, 0] / (xmax - xmin)

            # p_c = np.array([(p0[0] + p1[0]) / 2, (p0[1] + p2[1]) / 2, (p1_c_z + p2_c_z)/2])
            p_c = np.array([(p0[0] + p1[0]) / 2, (p0[1] + p2[1]) / 2, p1_c_z])
            p1_3d = calib.project_image_to_rect_single(p_c)

            ct_3d = p1_3d[:3] - bbox_3d[10]
            bbox_3d = (bbox_3d.T + ct_3d.reshape(3, 1)).T

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_1, line_thickness)

        elif clazz[0] == 0 and clazz[1] == 1 and clazz[2] == 0 and clazz[3] == 1:  # right and rightRear
            # 0------1------2
            # |      |      |
            # |      |      |
            # |      |      3
            # |      |   /
            # 4------5/

            x_center = (xmin + xmax) // 2 - dx
            p0 = (x_center - dw, ymin)
            p1 = (x_center + dw, ymin)
            p2 = (xmax, ymin + l0)
            p3 = (xmax, ymin + l0 + l1)
            p4 = (x_center - dw, ymin + l2)
            p5 = (x_center + dw, ymin + l3)

            p1_c, p1_c_z = np.array([(p0[0] + p4[0]) / 2, (p0[1] + p4[1]) / 2, 1]), h * P[1, 1] / l2
            p2_c, p2_c_z = np.array([(p1[0] + p5[0]) / 2, (p1[1] + p5[1]) / 2, 1]), h * P[1, 1] / l3
            p3_c, p3_c_z = np.array([(p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2, 1]), h * P[1, 1] / l1

            c_z = (p2_c_z + p3_c_z) / 2
            c_x = (xmin + xmax) / 2 - res_x
            c_y = (ymin + ymax) / 2 - res_y
            # p_c = np.array([c_x, c_y, 1])
            # p_c_3d = np.dot(pinv_P, p_c.T * c_z)[:3]
            # p_c_3d = np.dot(pinv_P, p_c.T * center_3d[-1])[:3]

            p_c = np.array([c_x, c_y, c_z])
            p_c_3d = calib.project_image_to_rect_single(p_c)

            bbox_3d = (bbox_3d.T + p_c_3d.reshape(3, 1)).T

            cv2.line(img, p0, p1, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p5, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p5, p4, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p0, p4, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p2, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p2, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p3, p5, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            
        bbox_3d = calib.project_rect_to_velo(bbox_3d)
        top_view = draw_box3d_on_top(top_view, bbox_3d)
        
        c1, c2 = (xmin, ymin), (xmax, ymax)
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(1) / 3, thickness=1)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(1) / 3, thickness=1)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0], c1[1] - 2), 0, float(1) / 3, [0, 255, 0],
                    thickness=1, lineType=cv2.FONT_HERSHEY_SIMPLEX)


    return img, top_view

DEVICE = 'cuda'

config_file = '/home/boden/Dev/ws/ws/BDNet_3D/configs/train/bdnet_kitti_3d.py'
checkpoint_file = '/home/boden/Dev/ws/ws/BDNet_3D/work_dirs/20210701/epoch_200.pth'

DEMO_ANNO_PATH = '/home/boden/Dev/ws/BDPilotDataset/Kitti/val/annotations/annotation_normal.json'
DEMO_DIR = '/home/boden/Dev/ws/BDPilotDataset/Kitti/val/images'

IMG_LIST = os.listdir(DEMO_DIR)
# IMG_FOLDER_LIST.sort()
# random.shuffle(IMG_FOLDER_LIST)

threshold = 0.4
iou_threshold = 0.2
obj_list = ['car', 'van', 'truck']

config = mmcv.Config.fromfile(config_file)
height = config.height
width = config.width
# load config information
img_mean = config.img_norm_cfg['mean']
img_std = config.img_norm_cfg['std']
num_classes = len(config.classes)

# build the model from a config file and a checkpoint file
model = build_detector(config.model, test_cfg=config.test_cfg)
checkpoint = torch.load(checkpoint_file, map_location=torch.device('cuda'))

model.CLASSES = checkpoint['meta']['CLASSES']
state_dict = checkpoint['state_dict']
# strip prefix of state_dict
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
# load state_dict
if hasattr(model, 'module'):
    load_state_dict(model.module, state_dict, False, None)
else:
    load_state_dict(model, state_dict, False, None)

model.cfg = config  # save the config in the model for convenience
model.cuda()
model.eval()

with open(DEMO_ANNO_PATH, 'r') as f:
    anno_dict = json.loads(f.read())

for frame_id, DEMO_IMG in enumerate(IMG_LIST):
    anno = anno_dict[DEMO_IMG]
    calib = Calibration(os.path.join('/home/boden/Dev/ws/BDPilotDataset/Kitti', 'calib', DEMO_IMG.replace('png', 'txt')))
    DEMO_IMG = os.path.join(DEMO_DIR,DEMO_IMG)
    img = cv2.imread(DEMO_IMG)
    time1 = time.time()
    outputs = inference_detector(model, DEMO_IMG)
    img_height,img_width,_ = img.shape
    top_view_image = get_top_image()
    
    scale = np.array([1280/img_width,384/img_height,1280/img_width,384/img_height])
    dynamic_bbox_result = anchor_based_post_process(outputs['detect'], scale)
    img, top_view_image = display(dynamic_bbox_result, img, calib, top_view_image)
    
    for bbox in anno:
        top_view_image = draw_box3d_on_top(top_view_image, np.array(bbox), color=(255, 255, 255))
    
    cv2.line(top_view_image, (0, 0), (0, 499), color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    
    rate = 20
    for i in range(500//rate):
        cv2.circle(top_view_image, (int(0),int(i*rate)), 3, (0, 0, 255), 2)
        cv2.putText(top_view_image, f'{int((500 - i*rate)*0.2)}', (int(5),int(i*rate)), 0, float(1) / 3, [255, 255, 255],
                    thickness=1, lineType=cv2.FONT_HERSHEY_SIMPLEX)
    cv2.line(top_view_image, (0, 499), (300, 499), color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    rate = 30
    for i in range(300//rate):
        cv2.circle(top_view_image, (int(i*rate),int(499)), 3, (0, 0, 255), 2)
        cv2.putText(top_view_image, f'{int((i*rate - 300)*0.2)}', (int(i*rate),int(494)), 0, float(1) / 3, [255, 255, 255],
                    thickness=1, lineType=cv2.FONT_HERSHEY_SIMPLEX)
    
    resize_height = 500
    resize_width = img_width * 500/img_height
    img = cv2.resize(img, (int(resize_width), int(resize_height)))
    img = np.concatenate((img, top_view_image), axis=1)

    print(f"{frame_id}.jpg")
    cv2.imwrite(f'/home/boden/Dev/ws/ws/BDNet_3D/data/kitti_3d_200_float_v1/{frame_id}.jpg',img)