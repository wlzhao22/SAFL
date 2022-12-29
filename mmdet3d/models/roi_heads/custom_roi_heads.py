'''
Modified from CenterNet2: https://github.com/xingyizhou/CenterNet2/blob/master/detectron2/modeling/roi_heads/roi_heads.py
'''

import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
from mmdet3d.models.utils.utils_2d.instances import Instances
from mmdet3d.models.utils.utils_2d.boxes import *
from .poolers import *
from .fast_rcnn import *
from .custom_fast_rcnn import *
from mmdet3d.models.utils.utils_2d.matcher import *
from mmdet3d.models.builder import HEADS
# from mmdet3d.models.losses.losses_2d.track_loss import MultiPosCrossEntropyLoss
from mmdet3d.models.losses.losses_2d.contrastive_loss import SupConLoss


def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.
    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.
    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals: List[Instances]) -> List[Instances]:
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.
    Returns:
        proposals: only contains proposals with at least one visible keypoint.
    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.
    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = nonzero_tuple(selection)[0]
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    return ret


class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

@HEADS.register_module()
class CustomCascadeROIHeads(torch.nn.Module):
    def __init__(
        self,
        in_channels=[256, 256, 256],
        pooler_resolution=7,
        pooler_scales=(1.0/8, 1.0/16, 1.0/32),
        sampling_ratio=0,
        pooler_type='ROIAlignV2',
        cascade_bbox_reg_weights=[(10.0, 10.0, 5.0, 5.0),
                                    (20.0, 20.0, 10.0, 10.0),
                                    (30.0, 30.0, 15.0, 15.0)],
        cascade_ious=[0.5, 0.6, 0.7],
        num_classes=4,
        proposal_append_gt=True,
        mult_proposal_score=True,
        batch_size_per_image=512,
        positive_fraction=0.25,
        fc_dim=1024,
        embedding_dim=128, 
        temperature=0.1,
        contrast_iou_thres=0.5,
        is_head_requires_grad=True,
        is_predictor_requires_grad=True,
        mask_on=False,
        keypoint_on=False,
    ):
        super().__init__()
        self.mult_proposal_score = mult_proposal_score
        self.proposal_append_gt = proposal_append_gt
        self.keypoint_on = keypoint_on 
        self.mask_on = mask_on

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        
        box_heads, box_predictors, proposal_matchers, box_embedding_heads = [], [], [], []
        for match_iou, bbox_reg_weights, in_channel in zip(cascade_ious, cascade_bbox_reg_weights, in_channels):
            box_head = FastRCNNConvFCHead([in_channel, pooler_resolution, pooler_resolution], is_requires_grad=is_head_requires_grad,fc_dim=fc_dim)
            box_heads.append(box_head)
            box_predictors.append(
                CustomFastRCNNOutputLayers(
                    mult_proposal_score=mult_proposal_score,
                    input_shape = box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights),
                    num_classes=num_classes,
                    is_requires_grad=is_predictor_requires_grad,
                    criterion=SupConLoss(temperature, contrast_iou_thres)
                )
            )
            proposal_matchers.append(Matcher([match_iou], [0, 1], allow_low_quality_matches=False))
            box_embedding_heads.append(ContrastiveHead(fc_dim, embedding_dim).cuda())

        self.num_cascade_stages = len(box_heads)
        self.box_heads = nn.ModuleList(box_heads)
        self.box_predictors = nn.ModuleList(box_predictors)

        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_classes = num_classes

        self.proposal_matcher = Matcher([0.5],[0, 1],allow_low_quality_matches=False,)
        self.proposal_append_gt = proposal_append_gt
        self.proposal_matchers = proposal_matchers
        self.box_embedding_heads = box_embedding_heads

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        has_gt = gt_classes.numel() > 0
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            gt_classes[matched_labels == 0] = self.num_classes
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes)

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals: List[Instances], targets: List[Instances]):
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, proposals_per_image.proposal_boxes)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(matched_idxs, matched_labels, targets_per_image.gt_classes)
            assert gt_classes.numel() == 0 or (0 <= gt_classes.min() and gt_classes.max() <= self.num_classes)
            
            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if match_quality_matrix.shape[0] > 0:
                iou, _ = match_quality_matrix.max(dim=0)
                proposals_per_image.ious = iou[sampled_idxs]
                # print(targets_per_image.gt_boxes.tensor.shape, proposals_per_image.gt_classes.shape, proposals_per_image.proposal_boxes.tensor.shape, match_quality_matrix.shape, iou.shape, iou[sampled_idxs].shape)
            else:
                # print(matched_idxs, sampled_idxs, match_quality_matrix)
                proposals_per_image.ious = torch.zeros_like(gt_classes).float()

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
               
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets):
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, proposals_per_image.proposal_boxes)  # shape=(n_gt, n_dt)
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            if len(targets_per_image) > 0:
                if match_quality_matrix.shape[0] > 0:
                    dt_idx = list(range(matched_idxs.shape[0]))
                    gt_ious = match_quality_matrix[matched_idxs, dt_idx] 
                else:
                    gt_ious = matched_idxs.new_zeros(matched_idxs.shape[0]).float()
                    
                gt_classes = targets_per_image.gt_classes[matched_idxs]            
                assert gt_classes.numel() == 0 or 0 <= gt_classes.min() and gt_classes.max() <= self.num_classes
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                assert gt_classes.shape == proposal_labels.shape
                gt_classes[proposal_labels == 0] = self.num_classes
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
                gt_ious = matched_idxs.new_zeros(matched_idxs.shape[0]).float()

            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes
            proposals_per_image.ious = gt_ious

        return proposals

    def _run_stage(self, features, proposals, stage):
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_heads[stage](box_features)
        predictions = self.box_predictors[stage](box_features)
        embeddings = self.box_embedding_heads[stage](box_features)
        return predictions, embeddings

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            if self.training:
                boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals

    def _forward_box(self, features, proposals, targets=None):
        if (not self.training) and self.mult_proposal_score:
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [
                    p.get('scores') for p in proposals]
            else:
                proposal_scores = [
                    p.get('objectness_logits') for p in proposals]
        
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        features = [f for f in features]
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                if self.training:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            predictions, _ = self._run_stage(features, proposals, k)
            prev_pred_boxes = self.box_predictors[k].predict_boxes(predictions, proposals)
            head_outputs.append((self.box_predictors[k], predictions, proposals))

        if self.training:
            losses = {}
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                stage_losses = predictor.losses(predictions, proposals)
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
        
            return losses
        else:            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            
            if self.mult_proposal_score:
                scores = [(s * ps[:, None]) ** 0.5 \
                    for s, ps in zip(scores, proposal_scores)]

            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            return pred_instances

    def _forward_mask(self, features, proposals, targets=None):
        # Not implemented yet.
        return {}

    def _forward_keypoint(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the keypoint prediction branch.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = dict([(f, features[f]) for f in self.keypoint_in_features])
        return self.keypoint_head(features, instances)

    def embedding_loss(self, gt_classes, gt_classes_ref, pred_embeddings, pred_embeddings_ref):
        row = gt_classes.shape[0]
        col = gt_classes_ref.shape[0]
        match = gt_classes.view(row, 1).repeat(1, col) == gt_classes_ref.view(1, col).repeat(row, 1)
        targets = torch.ones_like(match).cuda()*(-1)
        targets[match] = 1

        similarity = torch.mm(pred_embeddings, pred_embeddings_ref.t())
        similarity = torch.clamp(similarity, min=-1+1e-4, max=1 - 1e-4)
        
        weights = (match.sum(dim=1) > 0).float() + 1e-4
        similarity_loss = self.loss_track(similarity, targets, weights, avg_factor=weights.sum())
        return similarity_loss

    def forward(self, features, proposals):
        pred_instances = self._forward_box(features, proposals)
        return pred_instances
    
    def forward_train(self, inputs, outputs):
        features, proposals, targets = outputs['features'], outputs['proposals'], inputs['targets']
        proposals = self.label_and_sample_proposals(proposals, targets)
        losses = self._forward_box(features, proposals, targets)
        # Usually the original proposals used by the box head are used by the mask, keypoint
        # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
        # predicted by the box head.
        losses.update(self._forward_mask(features, proposals))
        losses.update(self._forward_keypoint(features, proposals))
        return losses

    def to_kitti_format(self, ret, img_metas, device='cpu', **kwargs):
        ret2 = []
        for r in ret:
            boxes = r.get('pred_boxes').tensor
            scores = r.get('scores')
            labels = r.get('pred_classes')

            r2 = [[] for _ in range(labels.max().item() + 1)] if labels.numel() > 0 else []
            for box, score, label in zip(boxes, scores, labels):
                r2[label.item()].append(torch.cat((box, score.unsqueeze(0))))
            for i in range(len(r2)):
                r2[i] = torch.stack(r2[i]) if len(r2[i]) > 0 else torch.empty((0,5,))
                r2[i] = r2[i].to(device)
            ret2.append(r2)
        return ret2
