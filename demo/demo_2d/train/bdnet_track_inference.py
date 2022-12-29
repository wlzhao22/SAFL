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
from mmdet.models.utils.efficientdet_utils import plot_one_box, get_index_label, STANDARD_COLORS, standard_to_bgr
from mmdet.models.utils.tracker.multitracker import JDETracker
from mmdet.models.utils.detect_post_process import simple_nms
from mmdet.models.utils.efficientdet_utils import _topk, tranpose_and_gather_feat, nms_torch, plot_one_box



DEVICE = 'cuda'

config_file = '/home/boden/Dev/ws/ws/BDNet_tracking/configs/bdnet/bdnet_tracking.py'
checkpoint_file = '/home/boden/Dev/ws/ws/BDNet_tracking/work_dirs/20210611/epoch_56.pth'

threshold = 0.4
iou_threshold = 0.2
obj_list = ['Sedan', 'Van', 'STruck', 'MTruck', 'LTruck', 'Bus','Traffic_Cone','Traffic_Bar','Traffic_Barrier']

color_list = standard_to_bgr(STANDARD_COLORS)
scale = torch.tensor([896/1418,384/640,896/1418,384/640],dtype=torch.float32).cuda()
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


def anchor_free_bbox_post_process(pred_heatmap, pred_wh, pred_off2d, scale):
    scale = torch.from_numpy(scale).cuda()
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
    center_list = []
    score_thr = 0.5
    for det in det_list:
        bbox, label, score = det['bbox'], det['label'], det['score']
        if score > score_thr:
            x1, y1, x2, y2 = bbox
            center_list.append(np.array([(x1+x2)/2, (y1+y2)/2]))
            x1 /= scale[0]
            y1 /= scale[1]
            x2 /= scale[0]
            y2 /= scale[1]
            result_list.append(np.array([x1, y1, x2, y2, score, label]))

    return np.array(result_list), np.array(center_list)

def anchor_base_feature_post_process(feature_heatmap, center_list):
    embedding_list = []
    if len(center_list) > 0:
        feature_heatmap = feature_heatmap.squeeze().permute(1,2,0).contiguous().cpu().numpy()
        center_list = center_list.astype(np.int)

        ct_int_x = center_list[:, 0]//8
        ct_int_y = center_list[:, 1]//8
        ct_res_x = center_list[:, 0]/8 - ct_int_x
        ct_res_y = center_list[:, 1]/8 - ct_int_y

        for x, y, res_x, res_y in zip(ct_int_x, ct_int_y, ct_res_x, ct_res_y):
            embedding = (1-res_x)*(1-res_y)*feature_heatmap[y, x] + (1-res_x)*(res_y)*feature_heatmap[y+1, x] \
                + (res_x)*(1-res_y)*feature_heatmap[y, x+1] + (res_x)*(res_y)*feature_heatmap[y+1, x+1]

            embedding_list.append(embedding)

    embedding_list = np.array(embedding_list)
    
    return embedding_list

def display(bbox, img, idx, imshow=False, imwrite=False,color=(0,0,0)):

    if bbox.shape[0] == 0:
        return img

    img = img.copy()

    for j in range(len(bbox)):
        x1, y1, x2, y2 = bbox[j,:4].astype(np.int)
        obj = obj_list[int(bbox[j,5])]
        score = float(bbox[j,4])
        plot_one_box(img, [x1, y1, x2, y2], label=obj,score=score,color=color)


    if imshow:
        cv2.imshow('img', img)
        cv2.waitKey(0)

    if imwrite:
        # cv2.imwrite(f'test/img_inferred_d{0}_this_repo_{i}.jpg', imgs[i])
        cv2.imwrite('/home/boden/Dev/ws/whz/BDPilot-merge/data/val/test_%d.jpg'%idx,img)

    return img

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 3200.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=2)
    return im

def display_bbox(bbox, img, start_index, color=(0,0,0)):
    if bbox.shape[0] == 0:
        return img

    for j in range(len(bbox)):
        x1, y1, x2, y2 = bbox[j,:4].astype(np.int)
        score = float(bbox[j,4])
        obj = obj_list[start_index + int(bbox[j,5])]
        plot_one_box(img, [x1, y1, x2, y2], label=obj,score=score,color=color)

    return img


# DEMO_DIR = '/home/boden/Dev/ws/whz/BDPilot-merge/data/validation_crop'
# DEMO_DIR = '/home/boden/Dev/ws/BDPilotDataset/Depth/save_00_calib_480'
# DEMO_DIR = '/home/boden/Dev/ws/BDPilotDataset/BDMerge/val_lane/scene_1'
DEMO_DIR = '/home/boden/Dev/ws/BDPilotDataset/BDMerge/val_lane/scene_2'
OUTPUT_PATH = '/home/boden/Dev/ws/ws/BDNet_tracking/data/output_0612_3'

IMG_LIST = os.listdir(DEMO_DIR)
IMG_LIST.sort()

tracker = JDETracker(conf_thres=0.3,mean=img_mean,std=img_std,num_classes=len(model.CLASSES))
scale = np.array([768/1418,384/640,768/1418,384/640])

for frame_id,DEMO_IMG in enumerate(IMG_LIST):
    img_name = DEMO_IMG
    DEMO_IMG = os.path.join(DEMO_DIR,DEMO_IMG)
    img = cv2.imread(DEMO_IMG)
    time1 = time.time()
    outputs = inference_detector(model, DEMO_IMG)
    dynamic_bbox_result, center_bbox_result = anchor_based_bbox_post_process(outputs['detect'], scale)
    static_bbox_result = anchor_free_bbox_post_process(outputs['static_detect']['hm'], outputs['static_detect']['wh'], outputs['static_detect']['off_2d'], scale)
    feature_bbox_result = anchor_base_feature_post_process(outputs['track']['embedding'], center_bbox_result)

    online_targets = tracker.update(dynamic_bbox_result, feature_bbox_result)
    online_tlwhs = []
    online_ids = []
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        online_tlwhs.append(tlwh)
        online_ids.append(tid)
    img = plot_tracking(img, online_tlwhs, online_ids, frame_id=frame_id, fps=1/(time.time()-time1))

    img = display_bbox(static_bbox_result, img, 6, color=(18,87,220))

    # for bbox in dynamic_bbox_result:
    #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), thickness=2)

    cv2.imwrite(os.path.join(OUTPUT_PATH, img_name), img)