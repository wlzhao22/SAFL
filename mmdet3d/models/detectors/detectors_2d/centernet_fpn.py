from ...builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from ...utils.utils_2d.key_config import *
import matplotlib.pyplot as plt 
import numpy as np

 
@DETECTORS.register_module()
class CenterNetFPN(BaseDetector):
    """Base class for tracking detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck and tracking instances based on the embeddings of output.

    """

    def __init__(self,
                 collect_keys=["img", "img_metas", "gt_bboxes_2d", \
                               "gt_bboxes_3d", "gt_labels", "gt_reids", \
                               "calib", "img_aug", "img_pre", "img_post", \
                               "car_mask", "inv_calib", "gt_lane_seg", "gt_lane_exist"],
                 backbones=None,
                 necks=None,
                 headers=None,
                 roi_headers=None,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None,
                 feature_layer=3):
        super(CenterNetFPN, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.collect_keys = collect_keys
        self.backbones, self.necks, self.headers, self.roi_headers = backbones, necks, headers, roi_headers
        self.pretrained = pretrained
        self.feature_layer = feature_layer

        self.init_nets()

    def init_nets(self):
        for k, v in self.backbones.items():
            if v is not None:
                bk = build_backbone(v)
                # if self.pretrained is not None:
                #     bk.load_pretrain(self.pretrained, k)
                setattr(self, k, bk)

        if self.necks is not None:
            for k, v in self.necks.items():
                if v is not None:
                    n = build_neck(v)
                    # if self.pretrained is not None:
                    #     n.load_pretrain(self.pretrained, k)
                    setattr(self, k, n)

        if self.headers is not None:
            for k, v in self.headers.items():
                if v is not None:
                    h = build_head(v)
                    # if self.pretrained is not None:
                    #     h.load_pretrain(self.pretrained, k)
                    setattr(self, k, h)

        if self.roi_headers is not None:
            for k, v in self.roi_headers.items():
                if v is not None:
                    rh = build_head(v)
                    # if self.pretrained is not None:
                    #     rh.load_pretrain(self.pretrained, k)
                    setattr(self, k, rh)

    def init_inputs(self, imgs, img_metas, para):
        inputs = {}
        inputs[COLLECTION_IMG] = imgs
        inputs[COLLECTION_IMG_METAS] = img_metas
        for k, v in para.items():
            inputs[k] = v
        return inputs

    def init_outputs(self, inputs):
        img = inputs['img']
        features = self.extract_feat(img)
        return {'features': features}

    def compute_losses(self, inputs, outputs):
        losses = {}
        if self.headers is not None: 
            for k, v in self.headers.items():
                if v is not None:
                    loss = getattr(self, k).forward_train(inputs, outputs)
                    losses.update(loss)
        
        if self.roi_headers is not None: 
            for k, v in self.roi_headers.items():
                if v is not None:
                    loss = getattr(self, k).forward_train(inputs, outputs)
                    losses.update(loss)
        return losses

    def forward_dummy(self, img):
        outputs = {}
        features = self.extract_feat(img)
        outputs['detect'] = self.centernet_head(features)
        return outputs

    def forward_train(self, imgs, img_metas, **kwargs):
        inputs = self.init_inputs(imgs, img_metas, kwargs)
        outputs = self.init_outputs(inputs)
        losses = self.compute_losses(inputs, outputs)
        return losses

    def simple_test(self, img, img_metas, rescale=False, debug=False):
        batch_size = len(img_metas)
        assert batch_size == 1
        features = self.extract_feat(img)
        if self.roi_headers is None:
            ret = self.centernet_head(features)
            ret = self.centernet_head.to_kitti_format(ret, img_metas=img_metas)
        else:
            rpn_res = self.centernet_head(features)
            ret = self.cascade_roi_head(features, rpn_res['proposals'])
            ret = self.cascade_roi_head.to_kitti_format(ret, img_metas=img_metas)
        return ret

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        raise NotImplementedError

    def extract_feat(self, img):
        return self.bifpn(self.feature_backbone(img))[:self.feature_layer]

    def show_result(
        self, 
        img, 
        result, 
        img_metas, 
        gt2d=None,
        gt3d=None,
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
        plt.imshow(img)
        plt.title('Image')

        # assert ori_shape == img_metas['ori_shape']
        img_shape = img_metas['img_shape']
        x_scale = 1.
        y_scale = 1.

        # Draw boxes 
        for label, boxes in enumerate(result):
            n = boxes.shape[0]
            for i in range(n):
                box = result[label][i][:4]
                score = result[label][i][4]
                if score < score_thr: continue 
                l, t, r, b = box 
                l, r = l * x_scale, r * x_scale
                t, b = t * y_scale, b * y_scale 
                _plot_ltrb(plt, (l, t, r, b), linewidth=1, c='r')
                plt.text(l, t, s='{};{:.3f}'.format(label, score), c='red')


def _plot_ltrb(ax, ltrb, **kwargs):
    l, t, r, b = ltrb 
    points = np.array([[l, t], [r, t], [r, b], [l, b], [l, t]])
    for i in range(4):
        line = points[i:i+2, :]
        ax.plot(line[:, 0], line[:, 1], **kwargs)
    