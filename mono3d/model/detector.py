import torch
from torch import Tensor, det
from torch import nn

from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_backbone, build_head, build_neck
from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
from mmcv.runner import auto_fp16
from mmdet.models.detectors.base import BaseDetector
from collections import OrderedDict, defaultdict
import torch.distributed as dist
import matplotlib.pyplot as plt 
import numpy as np
import time 


@DETECTORS.register_module()
class MonoFlex(BaseDetector):
    def __init__(self, backbone_cfg, neck_cfg, head_cfg, train_cfg, test_cfg, feature_index=-1, pretrained=False):
        super().__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.backbone = build_backbone(backbone_cfg)
        self.heads = build_head(head_cfg)
        self.neck = build_neck(neck_cfg) if neck_cfg is not None else None 
        self.feature_index = feature_index

    def init_inputs(self, img, img_metas, test=False, **para):
        inputs = {}
        inputs['img'] = img
        inputs['img_metas'] = img_metas
        for k, v in para.items():
            if isinstance(v, list) and test and k in ('edge_len', 'edge_indices', 'anchor_points'):
                v = v[0]
            inputs[k] = v
        return inputs

    def init_outputs(self, inputs, **kwargs):
        img = inputs['img']
        if 'outputs' not in kwargs:
            outputs = {}
            outputs['feature'] = self.extract_feat(img)
        else:
            outputs = kwargs['outputs']
        return outputs

    def extract_feat(self, img):
        featuremap = self.backbone(img)
        return featuremap
    
    def forward_train(self, imgs, img_metas, **kwargs):
        inputs = self.init_inputs(imgs, img_metas, **kwargs)
        outputs = self.init_outputs(inputs)
        losses = self.compute_losses(inputs, outputs, **kwargs)
        return losses

    def simple_test(self, imgs, img_metas, rescale=False, output_format='kitti', out_coord_system='lidar', depth_disturb=False, output_device='cpu', debug=False, **kwargs):
        t2 = time.time()

        # t = time.time()
        inputs = self.init_inputs(imgs, img_metas, test=True, **kwargs)
        # print("init inputs: ", time.time() - t)
        # t = time.time()
        outputs = self.init_outputs(inputs, **kwargs)
        # print("init outputs: ", time.time() - t)
        # t = time.time()
        result, eval_utils, visualize_preds = self.heads.forward_test(inputs, outputs, depth_disturb=depth_disturb, debug=debug)
        # print("head: ", time.time() - t)
        # t = time.time()
        if output_format == 'raw': 
            return result, eval_utils, visualize_preds
        ret = self.heads.to_kitti_format((result, eval_utils, visualize_preds), img_metas=img_metas, device=result.device, out_coord_system=out_coord_system, debug=debug, **kwargs)
        
        ret = [self.heads.nms_3d(item, result.device, out_device=output_device) for item in ret]

        # # TODO: remove this 
        # ret = [self.heads.result_sampling(item, img_metas, depth_thresh=20, score_rescale=True) for item in ret]

        # print('forward time: ', time.time() - t2)
        if output_format == 'kitti': 
            return ret 
        else:
            raise NotImplementedError()

    def aug_test(self, imgs, img_metas, rescale=False, return_loss=False, debug=False, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError()
        # dict of list to list of dict 
        kwargs_list = []
        for i in range(len(img_metas)):
            kwargs_list.append({})
            for k in kwargs and k not in ['debug', 'return_loss']:
                kwargs_list[i][k] = kwargs[k][i]

        ret = []
        for i, (imgs_, img_metas_) in enumerate(zip(imgs, img_metas)):
            result, eval_util, visualize_pred = \
                self.simple_test(
                    imgs_, img_metas_, rescale=rescale, return_loss=return_loss, debug=debug, 
                    output_format='raw',
                    **kwargs_list[i]
                )
            ret.append((result, eval_util, visualize_pred))
        
        # cat all outputs 
        results = [item[0] for item in ret]
        eval_utils = [item[1] for item in ret]
        visualize_preds = [item[2] for item in ret]
        def convert(d):
            d2 = defaultdict(list)
            d2.update({key: [] for key in d[0].keys()})
            for item in d: 
                for k, v in item.items():
                    d2[k].append(v) 
            for k in d2:
                if isinstance(d2[k][0], Tensor):
                    d2[k] = torch.cat(d2[k], axis=0)
                elif d2[k][0] is None:
                    assert all([it is None for it in d2[k]]), (k, d2[k])
                    continue
                else: 
                    raise TypeError(type(d2[k][0]))
            return d2 
        results = torch.cat(results, axis=0)
        eval_utils = convert(eval_utils)
        visualize_preds = convert(visualize_preds)

        results, eval_utils, visualize_preds = self.heads.nms_3d(results, eval_utils, visualize_preds)
        ret = self.heads.to_kitti_format((results, eval_utils, visualize_preds), img_metas[0], 'cpu')
        return ret 

    def compute_losses(self, inputs, outputs, **kwargs):
        loss_dict, log_loss_dict = self.heads.forward_train(inputs, outputs)
        return loss_dict, log_loss_dict
    
    # override
    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        losses, log_vars = losses 
        loss = sum(_value for _key, _value in losses.items()
                   if 'loss' in _key)
        assert 'loss' not in log_vars
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def show_result(
        self, 
        img, 
        result, 
        img_metas, 
        gt2d=None,
        gt3d=None,
        gt_occluded=None,
        score_thr=0.3,
        bbox_color='green',
        text_color='green',
        thickness=1,
        font_scale=0.5,
        win_name='',
        gt_text=False,
        figsize=(10, 10)
    ):
        assert len(img_metas) == len(result) == 1
        img_metas = img_metas[0]
        result = result[0]
        calib = img_metas['calib']
        P = calib['P2']  

        img = img[:, :, ::-1]
        plt.figure(figsize=figsize)
        plt.subplot(221)
        plt.imshow(img)
        plt.title('Image')

        # assert ori_shape == img_metas['ori_shape']
        img_shape = img_metas['img_shape']
        x_scale = 1.
        y_scale = 1.

        # Draw boxes 
        n = result['boxes'].shape[0]
        for i in range(n):
            box = result['boxes'][i]
            score = result['scores_3d'][i]
            if score < score_thr: continue 
            l, t, r, b = box 
            label = result['labels_3d'][i]
            l, r = l * x_scale, r * x_scale
            t, b = t * y_scale, b * y_scale 
            _plot_ltrb(plt, (l, t, r, b), linewidth=1, c='r')
            plt.text(l, t, s='{};{:.3f}'.format(label, score), c='red')
        if gt2d is not None:
            if isinstance(gt2d, Tensor): gt2d = gt2d.cpu()
            for i, box in enumerate(gt2d):
                l, t, r, b = box 
                l, r = l * x_scale, r * x_scale
                t, b = t * y_scale, b * y_scale 
                oc = gt_occluded[i] if gt_occluded is not None else None 
                c = 'b' if oc is None or oc == 0 else \
                    'g' if oc == 1 else \
                    'r' if oc == 2 else \
                    'k'
                _plot_ltrb(plt, (l, t, r, b), linewidth=1, c=c)
                if gt_text: plt.text(l, t, f'{i}', c=c)
        plt.subplot(222)
        plt.imshow(img)
        boxes_3d = result['boxes_3d']
        boxes_3d_cam = boxes_3d.convert_to(Box3DMode.CAM, calib['R0_rect'] @ calib['Tr_velo_to_cam'])
        n_gt = gt3d.tensor.shape[0] if gt3d else 0
        if n_gt > 0:
            gt3d_cam = gt3d.convert_to(Box3DMode.CAM, calib['R0_rect'] @ calib['Tr_velo_to_cam'])
        for i in range(n):
            score = result['scores_3d'][i]
            if score < score_thr: continue 
            _plot_box3d_cam(plt, boxes_3d_cam.tensor[i], P, c='r')
        for i in range(n_gt):
            _plot_box3d_cam(plt, gt3d_cam.tensor[i], P, c='b', text= i if gt_text else None)
        plt.subplot(223)
        plt.imshow(img)
        for i in range(n):
            score = result['scores_3d'][i]
            if score < score_thr: continue 
            center = boxes_3d_cam.tensor[i][:3].detach().cpu().numpy()
            center = np.array([*center, 1.])
            center = center @ P.T
            center[:2] = center[:2] / center[2]
            plt.scatter(center[[0]], center[[1]], c='r', s=1, marker='x')
        for i in range(n_gt):
            center = gt3d.tensor[i][:3].detach().cpu().numpy()
            center = np.array([*center, 1.])
            center = center @ P.T
            center[:2] = center[:2] / center[2]
            plt.scatter(center[[0]], center[[1]], c='b', s=1, marker='+')            
        plt.title('3d Centers')
        plt.subplot(224)
        for i in range(n):
            score = result['scores_3d'][i]
            if score < score_thr: continue 
            _plot_box3d_bev(plt, boxes_3d_cam.tensor[i], c='r')
        for i in range(n_gt):
            _plot_box3d_bev(plt, gt3d_cam.tensor[i], c='b')
        plt.ylim(-40, 40)
        plt.axis('equal')
        plt.show() 
        plt.close()

        if 'visualize_preds' in result:
            colors = ['g', 'r', 'y', 'cyan', 'orange', 'fuchsia']
            lines = np.array([
                0, 1,    1, 2,    2, 3,    3, 0, 
                4, 5,    5, 6,    6, 7,    7, 4,
                0, 4,    1, 5,    2, 6,    3, 7,    8, 9,
                0, 8,    1, 8,    2, 8,    3, 8,    
                4, 9,    5, 9,    6, 9,    7, 9
            ]).reshape((-1, 2))
            down_ratio = self.heads.down_ratio
            visualize_preds = result['visualize_preds']
            plt.figure(figsize=(12, 8))
            plt.subplot(121)
            plt.imshow(img)
            keypoints = visualize_preds['keypoints'].detach().cpu()     # shape: n, 10, 2
            proj_center = visualize_preds['proj_center'].detach().cpu() # shape: n, 2
            n = proj_center.shape[0]
            if n > 0:
                keypoints = (keypoints + proj_center.view(n, 1, 2)) * down_ratio
            for obj_id in range(keypoints.shape[0]):
                color = colors[obj_id % len(colors)]
                plt.scatter(keypoints[obj_id, :, 0], keypoints[obj_id, :, 1], c=color, s=2)
                for kpt_id in range(keypoints.shape[1]):
                    plt.text(x=keypoints[obj_id, kpt_id, 0], y=keypoints[obj_id, kpt_id, 1], s=str(kpt_id), c=color)
                for line in lines:
                    plt.plot(keypoints[obj_id, line, 0], keypoints[obj_id, line, 1], linewidth=1., c=color)
            plt.subplot(122)
            plt.imshow(img)
            heatmap = visualize_preds['heat_map'].detach().cpu()[0]
            if heatmap.shape[0] not in (1, 3):
                heatmap = heatmap.sum(axis=0, keep_dim=True)
            plt.imshow(heatmap.permute(1, 2, 0))
            if heatmap.shape[0] == 1:
                plt.colorbar()
            plt.show()
            plt.close()


def _plot_box3d_cam(ax, box3d:Tensor, P, text=False, **kwargs):
    '''
    @param box3d: a box from CameraInstance3DBoxes
    '''
    x, y, z, x_size, y_size, z_size, ry = box3d.detach().cpu().numpy()
    points = np.array([
        [-x_size / 2, y_size / 2, -z_size / 2, 1],
        [-x_size / 2, y_size / 2, +z_size / 2, 1], 
        [+x_size / 2, y_size / 2, +z_size / 2, 1],
        [+x_size / 2, y_size / 2, -z_size / 2, 1],
        [-x_size / 2, -y_size / 2, -z_size / 2, 1],
        [-x_size / 2, -y_size / 2, +z_size / 2, 1], 
        [+x_size / 2, -y_size / 2, +z_size / 2, 1],
        [+x_size / 2, -y_size / 2, -z_size / 2, 1],
    ])
    points[:, 1] -= y_size / 2
    rot = np.array([
        np.cos(-ry), 0, -np.sin(-ry),0,
        0,          1, 0,          0,
        np.sin(-ry), 0, np.cos(-ry), 0,
        0,          0, 0,          1
    ]).reshape((4, 4))
    points = points @ rot.T + np.array([x, y, z, 0])
    points = points @ P.T
    points = points[:, :2] / points[:, [2]]
    ax.scatter(points[:, 0], points[:, 1], s=1)
    lines = [
        [0, 1, 2, 3, 0],
        [4, 5, 6, 7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for line in lines: 
        pts = points[line]
        ax.plot(pts[:, 0], pts[:, 1], linewidth=0.5, **kwargs)
    if text is not None: ax.text(pts[0, 0], pts[0, 1], s=f'{text}', c='b')

def _plot_ltrb(ax, ltrb, **kwargs):
    l, t, r, b = ltrb 
    points = np.array([[l, t], [r, t], [r, b], [l, b], [l, t]])
    for i in range(4):
        line = points[i:i+2, :]
        ax.plot(line[:, 0], line[:, 1], **kwargs)
    
def _plot_box3d_bev(ax, box3d:Tensor, **kwargs):
    '''
    @param box3d: a box from CameraInstance3DBoxes
    '''
    x, y, z, x_size, y_size, z_size, ry = box3d.detach().cpu().numpy()
    points = np.array([
        [-x_size / 2, y_size / 2, -z_size / 2, 1],
        [-x_size / 2, y_size / 2, +z_size / 2, 1], 
        [+x_size / 2, y_size / 2, +z_size / 2, 1],
        [+x_size / 2, y_size / 2, -z_size / 2, 1],
        [-x_size / 2, -y_size / 2, -z_size / 2, 1],
        [-x_size / 2, -y_size / 2, +z_size / 2, 1], 
        [+x_size / 2, -y_size / 2, +z_size / 2, 1],
        [+x_size / 2, -y_size / 2, -z_size / 2, 1],
    ])
    rot = np.array([
        np.cos(-ry), 0, -np.sin(-ry),0,
        0,          1, 0,          0,
        np.sin(-ry), 0, np.cos(-ry), 0,
        0,          0, 0,          1
    ]).reshape((4, 4))
    points = points @ rot.T + np.array([x, y, z, 0])
    ax.scatter(points[:, 0], points[:, 2], s=1)
    lines = [0, 1, 2, 3, 0]
    ax.plot(points[lines][:, 0], points[lines][:, 2], linewidth=0.5, **kwargs)

