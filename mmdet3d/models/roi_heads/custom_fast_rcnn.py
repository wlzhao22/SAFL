import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
from mmdet3d.models.utils.utils_2d.instances import *
from mmdet3d.models.utils.utils_2d.boxes import *
from .poolers import *
from .fast_rcnn import *
from mmdet3d.models.utils.utils_2d.matcher import *
from .fast_rcnn import *
from torch.nn import functional as F
from mmdet3d.models.losses.losses_2d.centernet2_giou_loss import giou_loss
from mmdet3d.models.losses.losses_2d.centernet2_smooth_l1_loss import smooth_l1_loss


class CustomFastRCNNOutputs(FastRCNNOutputs):
    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        criterion,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        freq_weight=None,
    ):
        super().__init__(box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta, box_reg_loss_type)
        self._no_instances = (self.pred_class_logits.numel() == 0) or (len(proposals) == 0)
        self.criterion = criterion
        assert self.gt_classes.numel() == 0 or 0 <= self.gt_classes.min() <= self.gt_classes.max() < self.pred_class_logits.shape[-1]

    def softmax_cross_entropy_loss(self):
        """
        change _no_instance handling
        """
        if self._no_instances:
            return self.pred_class_logits.new_zeros([1])[0]
        else:
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def box_reg_loss(self):
        """
        change _no_instance handling and normalization
        """
        if self._no_instances:
            print('No instance in box reg loss')
            return self.pred_proposal_deltas.new_zeros([1])[0]

        box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
        if cls_agnostic_bbox_reg:
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            loss_box_reg = giou_loss(
                self._predict_boxes()[fg_inds[:, None], gt_class_cols],
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):
        loss_cls = self.softmax_cross_entropy_loss()
        return {
            "loss_cls": loss_cls, 
            "loss_box_reg": self.box_reg_loss(),
        }
        
    def predict_probs(self):
        """
        Deprecated
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)


class CustomFastRCNNOutputLayers(FastRCNNOutputLayers):
    def __init__(self, 
                mult_proposal_score, 
                input_shape,
                box2box_transform,
                num_classes: int,
                criterion,
                test_score_thresh: float = 0.05,
                test_nms_thresh: float = 0.7,
                test_topk_per_image: int = 100,
                cls_agnostic_bbox_reg: bool = True,
                smooth_l1_beta: float = 0.0,
                box_reg_loss_type: str = "smooth_l1",
                loss_weight: Union[float, Dict[str, float]] = 1.0,
                is_requires_grad = True):
        super().__init__(input_shape, box2box_transform, num_classes, test_score_thresh, test_nms_thresh, test_topk_per_image, cls_agnostic_bbox_reg, smooth_l1_beta, box_reg_loss_type, loss_weight)
        self.mult_proposal_score = mult_proposal_score
        self.criterion = criterion
        if not is_requires_grad:
            for p in self.parameters():
                p.requires_grad = False
            # self.eval()

    def losses(self, predictions, proposals):
        """
        enable advanced loss
        """
        scores, proposal_deltas = predictions
        losses = CustomFastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.criterion,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            None, 
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def inference(self, predictions, proposals):
        """
        enable use proposal boxes
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        if self.mult_proposal_score:
            proposal_scores = [p.get('objectness_logits') for p in proposals]
            scores = [(s * ps[:, None]) ** 0.5 for s, ps in zip(scores, proposal_scores)]
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )


    def predict_probs(self, predictions, proposals):
        """
        support sigmoid
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)



class ContrastiveHead(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, dim_in, feat_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )

    def forward(self, x):
        feat = self.head(x)
        feat_normalized = F.normalize(feat, dim=1)
        return feat_normalized