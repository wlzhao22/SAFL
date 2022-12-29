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
from mmdet.models.utils.efficientdet_utils import anchor_based_post_process,plot_one_box,get_index_label,STANDARD_COLORS,\
                                                    standard_to_bgr,anchor_free_post_process
from mmdet.models.utils.tracker.multitracker import JDETracker


DEVICE = 'cuda'

config_file = '/home/boden/Dev/ws/whz/BDPilot-repvgg/configs/bdnet/bdnet_fusion.py'
checkpoint_file = '/home/boden/Dev/ws/whz/BDPilot-repvgg/work_dirs/20210323/test_multi/latest.pth'

threshold = 0.3
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

if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']

# strip prefix of state_dict
if list(checkpoint.keys())[0].startswith('module.'):
    checkpoint = {k[7:]: v for k, v in checkpoint.items()}

# load state_dict
if hasattr(model, 'module'):
    load_state_dict(model.module, checkpoint, False, None)
else:
    load_state_dict(model, checkpoint, False, None)

model.cfg = config  # save the config in the model for convenience
model.cuda()
model.eval()

def display(bbox, img, idx, imshow=False, imwrite=False):

    if bbox.shape[0] == 0:
        return img

    img = img.copy()

    for j in range(len(bbox)):
        x1, y1, x2, y2 = bbox[j,:4].astype(np.int)
        obj = obj_list[int(bbox[j,5])]
        score = float(bbox[j,4])
        plot_one_box(img, [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])


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

    text_scale = max(1, image.shape[1] / 1600.)
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


# DEMO_DIR = '/home/boden/Dev/ws/merge_test/BDPilot/data/HWP/val_img/00_cloudy/part_021_20201102'
DEMO_DIR = '/home/boden/Dev/ws/whz/BDPilot-merge/data/validation_crop'
IMG_LIST = os.listdir(DEMO_DIR)
IMG_LIST.sort()

# tracker = JDETracker(conf_thres=0.3,mean=img_mean,std=img_std,num_classes=len(model.CLASSES))

converted_weights = OrderedDict()
print("=================")
for name, module in model.named_modules():
    if hasattr(module, 'repvgg_convert'):
        kernel, bias = module.repvgg_convert()
        converted_weights[name + '.rbr_reparam.weight'] = kernel
        converted_weights[name + '.rbr_reparam.bias'] = bias
    else:
        for p_name, p_tensor in module.named_parameters():
            full_name = name + '.' + p_name
            if full_name not in converted_weights:
                converted_weights[full_name] = p_tensor.detach().cpu().numpy()
        for p_name, p_tensor in module.named_buffers():
            full_name = name + '.' + p_name
            if full_name not in converted_weights:
                converted_weights[full_name] = p_tensor.cpu().numpy()


deploy_config_file = '/home/boden/Dev/ws/whz/BDPilot-repvgg/configs/bdnet/bdnet_fusion_deploy.py'
deploy_config = mmcv.Config.fromfile(deploy_config_file)
converted_tensor_weights = OrderedDict()

# build the model from a config file and a checkpoint file
deploy_model = build_detector(deploy_config.model, test_cfg=deploy_config.test_cfg)
for name, param in deploy_model.named_parameters():
    print('deploy named_parameters: ', name, param.size(), np.mean(converted_weights[name]))
    param.data = torch.from_numpy(converted_weights[name]).float()

for name, param in deploy_model.named_buffers():
    print('deploy named_buffers: ', name, param.size(), np.mean(converted_weights[name]))
    param.data = torch.from_numpy(converted_weights[name]).float()

torch.save(deploy_model.state_dict(), '/home/boden/Dev/ws/whz/BDPilot-repvgg/work_dirs/repvgg_deploy.pth')