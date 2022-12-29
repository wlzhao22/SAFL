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
from mmdet.models.utils.kitti_utils import *
from mmdet.models.utils.kitti_vis import *
import json

DEVICE = 'cuda'

config_file = '/home/boden/Dev/ws/ws/BDPilot_yolo_3d/configs/bdnet/bdnet_fusion_bird_view.py'
checkpoint_file = '/home/boden/Dev/ws/ws/BDPilot_RepVGG_v3/work_dirs/20210415/epoch_80.pth'

DEMO_ANNO_PATH = '/home/boden/Dev/ws/BDPilotDataset/Kitti/val/annotations/annotation_normal.json'
DEMO_DIR = '/home/boden/Dev/ws/BDPilotDataset/Kitti/val/images'
# DEMO_DIR = '/home/boden/Dev/ws/BDPilotDataset/Kitti/train/images'
IMG_LIST = os.listdir(DEMO_DIR)
# IMG_FOLDER_LIST.sort()
# random.shuffle(IMG_FOLDER_LIST)

threshold = 0.4
iou_threshold = 0.2
obj_list = ['car', 'van', 'truck']

# scale = torch.tensor([896/1418,384/640,896/1418,384/640],dtype=torch.float32).cuda()
# scale = torch.tensor([768/1418,384/640,768/1418,384/640],dtype=torch.float32).cuda()

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

def anchor_free_post_process(pred_heatmap, pred_wh, pred_off2d, scale):
    down_ratio = 8
    batch, cat, height, width = pred_heatmap.size()
    pred_heatmap = pred_heatmap.detach().sigmoid_()
    wh = pred_wh.detach()
    off2d = pred_off2d.detach()

    # perform nms on heatmaps
    # heat = simple_nms(pred_heatmap)  # used maxpool to filter the max score
    heat = pred_heatmap
    topk = 100
    # (batch, topk)
    scores, inds, clses, ys, xs = _topk(heat, topk=topk)

    if pred_off2d is not None:
        off2d = tranpose_and_gather_feat(off2d, inds)
        off2d = off2d.view(batch, topk, 2)
        xs = (xs.view(batch, topk, 1) + off2d[:, :, 0:1]) * down_ratio
        ys = (ys.view(batch, topk, 1) + off2d[:, :, 1:2]) * down_ratio
    else:
        xs = xs.view(batch, topk, 1) * down_ratio
        ys = ys.view(batch, topk, 1) * down_ratio

    wh = tranpose_and_gather_feat(wh, inds)

    wh = wh.view(batch, topk, 2)
    wh = wh * down_ratio
    clses = clses.view(batch, topk, 1).float()
    scores = scores.view(batch, topk, 1)

    bboxes2d = torch.cat([xs - wh[..., [0]] / 2, ys - wh[..., [1]] / 2,
                          xs + wh[..., [0]] / 2, ys + wh[..., [1]] / 2], dim=2)

    result_list = []
    score_thr = 0.3
    for batch_i in range(bboxes2d.shape[0]):
        scores_per_img = scores[batch_i].squeeze()
        bboxes2d_per_img = bboxes2d[batch_i]
        labels_per_img = clses[batch_i]

        nms_idx = nms_torch(bboxes2d_per_img, scores_per_img, iou_threshold=0.5)

        bboxes2d_per_img = bboxes2d_per_img[nms_idx] / scale

        labels_per_img = labels_per_img[nms_idx]
        scores_per_img = scores_per_img[nms_idx]

        scores_keep = (scores_per_img > score_thr)

        scores_per_img = scores_per_img[scores_keep].unsqueeze(1)
        bboxes2d_per_img = bboxes2d_per_img[scores_keep]

        labels_per_img = labels_per_img[scores_keep]

        bboxes_per_img = torch.cat([bboxes2d_per_img, scores_per_img, labels_per_img], dim=1)
        result_list.append(bboxes_per_img)
    result_list = torch.stack(result_list, dim=0)
    result_list = result_list.squeeze(0)
    result_list = result_list.cpu().numpy()
    return result_list

def anchor_based_post_process(det_list, scale):
    result_list = []
    for det in det_list:
        bbox, label, score = det['bbox'], det['label'], det['score']
        x1, y1, x2, y2 = bbox
        x1 /= scale[0]
        y1 /= scale[1]
        x2 /= scale[0]
        y2 /= scale[1]

        shortx, h, w, l, head_score, head_label, rotation_res, rotation_center = det['shortx'], det['h'], det['w'], det['l'], det['head_score'], det['head_label'], det['rotation_res'], det['rotation_center'] 
        shortx /= scale[0]
        lf, rf, rr, lr = det['lf'], det['rf'], det['rr'], det['lr']
        lf /= scale[1]
        rf /= scale[1]
        rr /= scale[1]
        lr /= scale[1]

        rotation = rotation_center - np.arctan(rotation_res)
        if rotation > np.pi:
            rotation -= 2*np.pi
        elif rotation < -np.pi:
            rotation += 2*np.pi

        result_list.append(np.array([x1, y1, x2, y2, score, label, shortx, lf, rf, rr, lr, h, w, l, head_score, head_label, rotation]))

    return np.array(result_list)


def compute_box_3d(ry, l, w, h, P):
   
    R = roty(ry)
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))

    return corners_3d.T

def display(bbox, img, start_index, calib, top_view, color=(0,0,0), socre_threshold=0.4):
    if bbox.shape[0] == 0:
        return img, top_view

    P = calib.P
    ddd_threshold = 0.3
    color_0 = (0, 255, 255)
    color_1 = (101, 67, 254) # tail
    color_2 = (0, 208, 244)
    color_3 = (255, 255, 255) # front
    color_4 = (0, 0, 255) # box
    line_thickness=2
    pinv_P = np.linalg.pinv(P)
    for j in range(len(bbox)):
        is_draw_bbox = True
        x1, y1, x2, y2 = bbox[j,:4].astype(np.int)
        shortx = bbox[j, 6].astype(np.int)
        lf, rf, rr, lr = bbox[j, 7:11].astype(np.int)
        score = float(bbox[j,4])
        label = obj_list[start_index + int(bbox[j,5])]
        h, w, l = bbox[j, 11:14]
        rotation = bbox[j, 16]
        bbox_3d = compute_box_3d(rotation, l, w, h, P)

        if score < socre_threshold:
            continue

        if bbox[j, 14] > ddd_threshold and bbox[j, 15] == 0:
            # 0------1
            # |      |      
            # |      |      
            # |      |      
            # |      |      
            # 2rr----3rf

            is_draw_bbox = False

            p0 = (x1, y1)
            p1 = (x2, y1)
            p2 = (x1, y2)
            p3 = (x2, y2)

            p0_z = h * P[1,1] / rr 
            p0_3d = np.dot(pinv_P, np.array([x1, y1, 1]).T * p0_z)
            center_3d = p0_3d[:3] - bbox_3d[6]
            bbox_3d = (bbox_3d.T + center_3d.reshape(3, 1)).T

            cv2.line(img, p0, p1, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p0, p2, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p3, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p2, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
        
        elif bbox[j, 14] > ddd_threshold and bbox[j, 15] == 1:
            # 0------1------2
            # |      |      |
            # |      |      |
            # 3rr    |      |
            #    \   |      |
            #      \ 4rf----5 

            is_draw_bbox = False

            p0 = (x1, y1)
            p1 = (x2-shortx, y1)
            p2 = (x2, y1)
            p3 = (x1, y1+rr)
            p4 = (x2-shortx, y1 + rf)
            p5 = (x2, y1 + rf)

            if y1 + rf >= img.shape[0]:
                p0_z = h * P[1,1] / rr 
                p0_3d = np.dot(pinv_P, np.array([x1, y1, 1]).T * p0_z)
                center_3d = p0_3d[:3] - bbox_3d[6]
            else:
                p1_z = h * P[1,1] / rf 
                p1_3d = np.dot(pinv_P, np.array([x2-shortx, y1, 1]).T * p1_z)
                center_3d = p1_3d[:3] - bbox_3d[5]

            bbox_3d = (bbox_3d.T + center_3d.reshape(3, 1)).T
            cv2.line(img, p0, p1, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p2, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p0, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p4, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p2, p5, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p5, p4, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p4, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)

        elif bbox[j, 14] > ddd_threshold and bbox[j, 15] == 2:
            # 0------1
            # |      |      
            # |      |      
            # |      |      
            # |      |      
            # 2rf----3lf 

            is_draw_bbox = False
            p0 = (x1, y1)
            p1 = (x2, y1)
            p2 = (x1, y2)
            p3 = (x2, y2)

            p0_z = h * P[1,1] / rf
            p0_3d = np.dot(pinv_P, np.array([x1, y1, 1]).T * p0_z)
            center_3d = p0_3d[:3] - bbox_3d[5]
            bbox_3d = (bbox_3d.T + center_3d.reshape(3, 1)).T

            cv2.line(img, p0, p1, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p0, p2, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p3, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p2, p3, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
        
        elif bbox[j, 14] > ddd_threshold and bbox[j, 15] == 3:
            # 0------1------2
            # |      |      |
            # |      |      |
            # |      |      3lr
            # |      |     / 
            # 5------4lf /  
            
            is_draw_bbox = False

            p0 = (x1, y1)
            p1 = (x1+shortx, y1)
            p2 = (x2, y1)
            p3 = (x2, y1+lr)
            p4 = (x1+shortx, y1 + lf)
            p5 = (x1, y1 + lf)

            if y1 + lf >= img.shape[0]:
                p2_z = h * P[1,1] / lr 
                p2_3d = np.dot(pinv_P, np.array([x2, y1, 1]).T * p2_z)
                center_3d = p2_3d[:3] - bbox_3d[7]
            else:
                p1_z = h * P[1,1] / lf 
                p1_3d = np.dot(pinv_P, np.array([x1+shortx, y1, 1]).T * p1_z)
                center_3d = p1_3d[:3] - bbox_3d[4]

            bbox_3d = (bbox_3d.T + center_3d.reshape(3, 1)).T

            cv2.line(img, p0, p1, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p2, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p0, p5, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p4, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p2, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p5, p4, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p4, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
        
        elif bbox[j, 14] > ddd_threshold and bbox[j, 15] == 4:
            # 0------1
            # |      |      
            # |      |      
            # |      |      
            # |      |      
            # 2lf----3lr 
            
            is_draw_bbox = False

            p0 = (x1, y1)
            p1 = (x2, y1)
            p2 = (x1, y2)
            p3 = (x2, y2)
            
            p0_z = h * P[1,1] / lf
            p0_3d = np.dot(pinv_P, np.array([x1, y1, 1]).T * p0_z)
            center_3d = p0_3d[:3] - bbox_3d[4]
            bbox_3d = (bbox_3d.T + center_3d.reshape(3, 1)).T

            cv2.line(img, p0, p1, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p0, p2, color=color_3, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p3, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p2, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)

        elif bbox[j, 14] > ddd_threshold and bbox[j, 15] == 5:
            # 0------1------2
            # |      |      |
            # |      |      |
            # 3lf    |      |
            #    \   |      |
            #      \ 4lr----5 
           
            is_draw_bbox = False
           
            p0 = (x1, y1)
            p1 = (x2-shortx, y1)
            p2 = (x2, y1)
            p3 = (x1, y1+lf)
            p4 = (x2-shortx, y1+lr)
            p5 = (x2, y1+lr)

            if y1 + lr >= img.shape[0]:
                p0_z = h * P[1,1] / lf 
                p0_3d = np.dot(pinv_P, np.array([x1, y1, 1]).T * p0_z)
                center_3d = p0_3d[:3] - bbox_3d[4]
            else:
                p1_z = h * P[1,1] / lr 
                p1_3d = np.dot(pinv_P, np.array([x2-shortx, y1, 1]).T * p1_z)
                center_3d = p1_3d[:3] - bbox_3d[7]

            bbox_3d = (bbox_3d.T + center_3d.reshape(3, 1)).T

            cv2.line(img, p0, p1, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p2, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p0, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p4, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p2, p5, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p5, p4, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p4, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
        
        elif bbox[j, 14] > ddd_threshold and bbox[j, 15] == 6:
            # 0------1
            # |      |      
            # |      |      
            # |      |      
            # |      |      
            # 2lr----3rr 

            is_draw_bbox = False

            p0 = (x1, y1)
            p1 = (x2, y1)
            p2 = (x1, y2)
            p3 = (x2, y2)

            p0_z = h * P[1,1] / lr
            p0_3d = np.dot(pinv_P, np.array([x1, y1, 1]).T * p0_z)
            center_3d = p0_3d[:3] - bbox_3d[7]
            bbox_3d = (bbox_3d.T + center_3d.reshape(3, 1)).T

            cv2.line(img, p0, p1, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p0, p2, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p3, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p2, p3, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
        
        elif bbox[j, 14] > ddd_threshold and bbox[j, 15] == 7:
            # 0------1------2
            # |      |      |
            # |      |      |
            # |      |      3rf
            # |      |     / 
            # 5------4rr /  

            is_draw_bbox = False
            
            p0 = (x1, y1)
            p1 = (x1+shortx, y1)
            p2 = (x2, y1)
            p3 = (x2, y1+rf)
            p4 = (x1+shortx, y1+rr)
            p5 = (x1, y1+rr)

            if y1 + rr >= img.shape[0]:
                p2_z = h * P[1,1] / rf 
                p2_3d = np.dot(pinv_P, np.array([x2, y1, 1]).T * p2_z)
                center_3d = p2_3d[:3] - bbox_3d[5]
            else:
                p1_z = h * P[1,1] / rr 
                p1_3d = np.dot(pinv_P, np.array([x1+shortx, y1, 1]).T * p1_z)
                center_3d = p1_3d[:3] - bbox_3d[6]

            bbox_3d = (bbox_3d.T + center_3d.reshape(3, 1)).T

            cv2.line(img, p0, p1, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p2, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p0, p5, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p1, p4, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p2, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p5, p4, color=color_1, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p4, p3, color=color_2, thickness=line_thickness, lineType=cv2.LINE_AA)
        
        bbox_3d = calib.project_rect_to_velo(bbox_3d)
        top_view = draw_box3d_on_top(top_view, bbox_3d)
        
        if is_draw_bbox:
            p0 = (x1, y1)
            p1 = (x2, y1)
            p2 = (x1, y2)
            p3 = (x2, y2)
            cv2.line(img, p0, p1, color=color_4, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p0, p2, color=color_4, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p2, p3, color=color_4, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p3, p1, color=color_4, thickness=line_thickness, lineType=cv2.LINE_AA)

        c1, c2 = (x1, y1), (x2, y2)
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(1) / 3, thickness=1)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(1) / 3, thickness=1)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0], c1[1] - 2), 0, float(1) / 3, [0, 255, 0],
                    thickness=1, lineType=cv2.FONT_HERSHEY_SIMPLEX)


    return img, top_view

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
    
    scale = np.array([768/img_width,384/img_height,768/img_width,384/img_height])
    dynamic_bbox_result = anchor_based_post_process(outputs['detect'], scale)
    img, top_view_image = display(dynamic_bbox_result, img, 0, calib, top_view_image, color=(0,255,127))
    
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
    cv2.imwrite(f'/home/boden/Dev/ws/ws/BDPilot_RepVGG_v3/data/kitti_ddd_val_epoch_80/{frame_id}.jpg',img)