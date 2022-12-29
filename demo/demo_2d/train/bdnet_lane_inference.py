from mmdet3d.apis import inference_2d_detector
from mmdet3d.models import build_detector
from mmcv.runner import load_state_dict

import mmcv
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
import shutil
from mmdet3d.models.utils.utils_2d.kitti_vis import *
from mmdet3d.models.utils.utils_2d.efficientdet_utils import simple_nms, _topk, tranpose_and_gather_feat, plot_one_box, nms_torch

fx = 987.16999908
fy = 992.77468844
cx = 952.14970514
cy = 634.91434312

rotationMatrix = np.array([[0.02013337, 0.01025601, 0.9997447],
                            [-0.99956417, -0.02138628, 0.02034913],
                            [0.02158952, -0.99971868, 0.00982096]])

translationMatrix = np.array([1.619, 0, 1.699])

innerMatrix = np.array([[fx, 0,  cx],
                        [0,  fy, cy],
                        [0,  0,  1] ])

croped_x = 247 
croped_y = 214 + 120

obj_list = ['Sedan', 'Van', 'STruck', 'MTruck', 'LTruck', 'Bus','Traffic_Cone','Traffic_Bar','Traffic_Barrier']

def anchor_free_bbox_post_process(pred_heatmap, pred_wh, pred_off2d, scale):
    scale = torch.from_numpy(scale).cuda()[[1,0,1,0]]
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

        bboxes_per_img = torch.cat([bboxes2d_per_img.float(), scores_per_img.float(), labels_per_img.float()], dim=1)
        result_list.append(bboxes_per_img)
    result_list = torch.stack(result_list, dim=0)
    result_list = result_list.squeeze(0)
    result_list = result_list.cpu().numpy()
    return result_list

def anchor_based_bbox_post_process(det_list, scale):
    result_list = []
    score_thr = 0.35
    for det in det_list:
        bbox, label, score = det['bbox'], det['label'], det['score']
        if score > score_thr:
            x1, y1, x2, y2 = bbox
            x1 /= scale[1]
            y1 /= scale[0]
            x2 /= scale[1]
            y2 /= scale[0]
            result_list.append(np.array([x1, y1, x2, y2, score, label]))

    return np.array(result_list)

# def freespace_post_process(seg, reg, scale):
#     reg = reg.squeeze().detach()
#     predict_mask = torch.argmax(seg.squeeze(1),dim=1)
#     predict_mask = predict_mask.detach().squeeze()
#     scale = torch.tensor([scale[0], scale[1]])
#     points = []
#     for i in range(predict_mask.shape[0]):
#         point = [int(i*4/scale[1]),int(((predict_mask[i] + reg[predict_mask[i], i])*4/scale[0]).round())]
#         points.append(point)
#     return points

def freespace_post_process(seg, scale):
    # predict_mask = torch.argmax(seg.squeeze(1),dim=1)
    # predict_mask = predict_mask.detach().squeeze()
    # scale = torch.tensor([scale[0], scale[1]])
    # points = []
    # for i in range(predict_mask.shape[0]):
    #     point = [int(i*8/scale[1]),int(((predict_mask[i])*8/scale[0]).round())]
    #     points.append(point)
    # return points
    predict_mask = torch.argmax(seg.squeeze(1),dim=1)
    predict_mask = predict_mask.detach().squeeze()
    scale = torch.tensor([scale[0], scale[1]])
    points = []
    for i in range(predict_mask.shape[0]):
        if i>0 and i<predict_mask.shape[0]-1:
            if predict_mask[i]<47:
                point = [int(i*8/scale[1]),int((((0.5*predict_mask[i-1]+2*predict_mask[i]+0.5*predict_mask[i+1]).float()*8/3)/scale[0]).round())]
                points.append(point)
        else:
            if predict_mask[i]<47:
                point = [int(i*8/scale[1]),int((predict_mask[i]*8/scale[0]).round())]
                points.append(point)
    return points

def center_cluster(center_list, thresh):
    counter = 1
    label_order = []
    centers_order = []
    while center_list.shape[0] > 0:
        cur_center = center_list[0]
        center_list = center_list[1:]
        centers_order.append(cur_center)
        label_order.append(counter)

        dis = (center_list[:, 0] - cur_center[0])**2 + (center_list[:, 1] - cur_center[1])**2
        for ct in center_list[dis <= thresh]:
            centers_order.append(ct)
            label_order.append(counter)

        center_list = center_list[dis > thresh]
        counter += 1
    
    cluster_result = np.concatenate((np.array(centers_order).reshape(-1, 3), (np.array(label_order).reshape(-1,1))), 1) 
    return cluster_result

def lane_post_process(exist_preds, segmentation_pred_xs, segmentation_pred_ys, regression_pred_xs, scale):
    
    exist_preds = torch.clamp(exist_preds.sigmoid_(), min=1e-4, max=1 - 1e-4)
    exist_preds = exist_preds.detach()
    topk = 10

    # exist_preds = simple_nms(exist_preds)
    batch, cat, height, width = exist_preds.size()
    scores, inds, clses, ys, xs = _topk(exist_preds, topk=topk)
    xs = xs.view(batch, topk, 1)
    ys = ys.view(batch, topk, 1)
    clses = clses.view(batch, topk, 1).float()
    scores = scores.view(batch, topk, 1)
    lane_centers = torch.cat([xs, ys], dim=2)
    lane_center_list = []
    score_thr = 0.4
    for batch_i in range(lane_centers.shape[0]):
        scores_per_img = scores[batch_i].squeeze()
        lane_centers_per_img = lane_centers[batch_i]

        scores_keep = (scores_per_img > score_thr)
        scores_per_img = scores_per_img[scores_keep].unsqueeze(1)
        lane_centers_per_img = lane_centers_per_img[scores_keep]

        lane_center_per_img = torch.cat([lane_centers_per_img, scores_per_img], dim=1)
        lane_center_list.append(lane_center_per_img)
    lane_center_list = torch.stack(lane_center_list, dim=0).squeeze(0).cpu().numpy()

    segmentation_pred_xs = F.softmax(segmentation_pred_xs, dim=1)
    segmentation_pred_xs = segmentation_pred_xs.permute(0,2,3,1)
    segmentation_pred_xs_score, _ = segmentation_pred_xs.max(dim=3)
    segmentation_pred_xs_score = segmentation_pred_xs_score.squeeze()
    segmentation_pred_xs_index = segmentation_pred_xs.argmax(dim=3).squeeze()

    segmentation_pred_ys = F.softmax(segmentation_pred_ys, dim=1)
    segmentation_pred_ys = segmentation_pred_ys.permute(0,2,3,1)
    segmentation_pred_ys_score, _ = segmentation_pred_ys.max(dim=3)
    segmentation_pred_ys_score = segmentation_pred_ys_score.squeeze()
    segmentation_pred_ys_index = segmentation_pred_ys.argmax(dim=3).squeeze()

    segmentation_pred_index = torch.stack((segmentation_pred_xs_index, segmentation_pred_ys_index), 2).cpu().numpy().reshape(-1, 2)

    label_map = {}
    label_xy = segmentation_pred_index[(segmentation_pred_index[:, 0] > 0) & (segmentation_pred_index[:, 1] > 0)]
    max_dis = 5
    centers = []
    
    lane_center_list = center_cluster(lane_center_list, max_dis)
    label_map_mask = {k:v+1 for v, k in enumerate(set(lane_center_list[np.argsort(lane_center_list[:, 0]), -1].tolist()))}
    for x, y in label_xy:
        min_dis = 99999
        x -= 1
        y -= 1
        for i, center in enumerate(lane_center_list):
            dis = (center[0] - x)**2 + (center[1] - y)**2
            if dis <= max_dis and dis < min_dis:
                label_map[x+48*y] = label_map_mask[center[3]]
                min_dis = dis
                if [x, y] not in centers:
                    centers.append([x, y])
    
    lane_result = []
    regression_pred_xs = regression_pred_xs.squeeze().permute(1,2,0).cpu().numpy()
    segmentation_pred_index = segmentation_pred_xs_index-1 + 48*(segmentation_pred_ys_index-1)
    labels = label_xy[:, 0]-1 + 48*(label_xy[:, 1]-1)
    label_map_2 = {k:v+1 for v, k in enumerate(set(labels.tolist()))}
    for i in range(segmentation_pred_index.shape[0]):
        for j in range(segmentation_pred_index.shape[1]):
            label = label_map.get(int(segmentation_pred_index[i, j]), None)
            if label:
                lane_result.append([int(j/scale[1]*8 + regression_pred_xs[i, j, 0]/scale[1]*8), int(i/scale[0]*8 + regression_pred_xs[i, j, 1]/scale[0]*8), label])
            
    return np.array(lane_result), label_map_2, lane_center_list

def display_bbox(bbox, img, start_index, color=(0,0,0)):
    if bbox.shape[0] == 0:
        return img

    for j in range(len(bbox)):
        x1, y1, x2, y2 = bbox[j,:4].astype(np.int)
        score = float(bbox[j,4])
        obj = obj_list[start_index + int(bbox[j,5])]
        plot_one_box(img, [x1, y1, x2, y2], label=obj,score=score,color=color)

    return img

# IMAGE_PATH = '/home/boden/Dev/ws/BDPilotDataset/BDMerge/images_crop_640'
# IMAGE_PATH = '/home/boden/Dev/ws/BDPilotDataset/BDMerge/val_images'
IMAGE_PATH = '/home/haozhe/Desktop/project/upload/0712/mmdetection3d/data/bdpilot/validation_crop'
OUTPUT_PATH = '/home/haozhe/Desktop/project/upload/0712/mmdetection3d/data/bdpilot/validation_crop_result'

config_file = '/home/haozhe/Desktop/project/upload/0712/mmdetection3d/configs/configs_2d/train/bdnet_lane.py'
checkpoint_file = '/home/haozhe/Desktop/project/upload/0712/mmdetection3d/work_dirs/epoch_106.pth'

config = mmcv.Config.fromfile(config_file)
num_class = 5
color = [(0, 191, 255), (118, 34, 34), (220,220,220), (128,128,128),(255,105,180),(173,216,230) ,(152,251,152) ,(255,218,185),(0, 0, 205), (123, 104, 238), (255, 160, 122), (107, 142, 35), (244, 164, 96), (255, 248, 220)]
# Build the model from a config file and a cheackpoint file
model = build_detector(config.model, test_cfg=config.test_cfg)
cheackpoint = torch.load(checkpoint_file, map_location=torch.device('cuda'))
model.CLASSES = cheackpoint['meta']['CLASSES']
state_dict = cheackpoint['state_dict']

# Strip prefix of state_dict
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k[7:]: v for k, v in cheackpoint['state_dict'].items()}

# Load state_dict
if hasattr(model, 'module'):
    load_state_dict(model.module, state_dict, False, None)
else:
    load_state_dict(model, state_dict, False, None)

model.cfg = config  # save the config in the model for convenience
model.cuda()
model.eval()


if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)
os.makedirs(OUTPUT_PATH)

threshold = 0.3
img_list = sorted(os.listdir(IMAGE_PATH))
for image_name in img_list:
    print(image_name)
    image_path = os.path.join(IMAGE_PATH, image_name)
    img = cv2.imread(image_path)
    res = inference_2d_detector(model, image_path)
    org_size = img.shape
    scale = np.array([384/img.shape[0], 768/img.shape[1]])

    lane_result, label_map, lane_center_list = lane_post_process(res['lane']['lane_exist'], res['lane']['lane_seg_x'], res['lane']['lane_seg_y'], res['lane']['lane_reg_xy'], scale)
    for i, item in enumerate(lane_result):
        x, y, label = item
        cv2.circle(img, (int(x), int(y)), 2, color[int(label-1)%14], 2)
    
    for i, item in enumerate(label_map.items()):
        k, v = item
        x = k%48
        y = k//48
        cv2.circle(img, (60*i+50, 50), 3, color[(v-1)%14], 3)
        cv2.putText(img,f"({x}, {y})",(60*i+50, 70), cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,255),1)
        cv2.circle(img, (int(x*16/scale[1]), int(y*16/scale[0])), 5, color[(v-1)%14], 5)

    for i, item in enumerate(lane_center_list):
        x, y = item[:2]
        cv2.circle(img, (int(x*16/scale[1]), int(y*16/scale[0])), 10, color[i%14], 10)

    # dynamic_bbox_result = anchor_based_bbox_post_process(res['detect'], scale)
    # img = display_bbox(dynamic_bbox_result, img, 0, color=(0,255,127))
    #
    # static_bbox_result = anchor_free_bbox_post_process(res['static_detect']['hm'], res['static_detect']['wh'], res['static_detect']['off_2d'], scale)
    # img = display_bbox(static_bbox_result, img, 6, color=(18,87,220))

    # points = freespace_post_process(res['free_space_pred']['freespace_seg'], res['free_space_pred']['freespace_reg'], scale)
    # points = freespace_post_process(res['free_space_pred']['free_space_pred'], scale)
    # for i in range(len(points)):
    #     cv2.circle(img, (points[i][0], points[i][1]), 1, color[-1], -1)
    #     cv2.line(img, (points[i][0], points[i][1]), (points[i][0], img.shape[0]-1), color=color[-1], thickness=1, lineType=cv2.LINE_AA)


    cv2.imwrite(os.path.join(OUTPUT_PATH, image_name), img)

