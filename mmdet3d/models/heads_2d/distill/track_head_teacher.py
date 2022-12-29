import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet3d.models.losses.losses_2d.track_loss import MultiPosCrossEntropyLoss, L2Loss
from mmdet3d.models.utils.utils_2d.key_config import *
from mmdet3d.models.builder import HEADS

class AnchorFreeModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AnchorFreeModule, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, U):
        f = self.conv(U)
        return f

@HEADS.register_module()
class TrackHeadTea(nn.Module):
    def __init__(self, in_channels, num_ids, feature_dim=128, positive_iou_threshold=0.7, negative_iou_threshold=0.3, down_ratio=8, num_cls=6):
        super(TrackHeadTea, self).__init__()
        self.num_cls = num_cls
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        self.down_ratio = down_ratio
        self.embedding_head = AnchorFreeModule(in_channels, feature_dim)
        self.loss_track = MultiPosCrossEntropyLoss(loss_weight=1)

    def forward_single(self, embeddings):
        outputs = {}
        outputs[OUTPUT_TEACHER_TRACK_EMBEDDING] = self.embedding_head(embeddings[0])

        return outputs
    
    def forward(self, embeddings):
        return self.forward_single(embeddings)
    
    def forward_train(self, inputs, outputs):
        outputs.update(self.forward_single(outputs[OUTPUT_TEACHER_FEATURE_AFFINE]))
        outputs[OUTPUT_TEACHER_TRACK_EMBEDDING_REF] = self.forward_single(outputs[OUTPUT_TEACHER_FEATURE_REF])[OUTPUT_TEACHER_TRACK_EMBEDDING]
        track_loss = self.loss(inputs, outputs)
        return track_loss
    
    def loss(self, inputs, outputs):
        gt_bboxes_2d, gt_labels, gt_reids = inputs[COLLECTION_GT_BBOX_2D], inputs[COLLECTION_GT_LABEL], inputs[COLLECTION_GT_REID]
        bbox_preds, embedding_preds = outputs[OUTPUT_DETECTION_YOLO_BOX_LIST].detach(), outputs[OUTPUT_TEACHER_TRACK_EMBEDDING]

        gt_bboxes_2d_ref, gt_labels_ref, gt_reids_ref = inputs[COLLECTION_GT_BBOX_2D_REF], inputs[COLLECTION_GT_LABEL_REF], inputs[COLLECTION_GT_REID_REF]
        bbox_preds_ref, embedding_preds_ref = outputs[OUTPUT_DETECTION_YOLO_BOX_LIST_REF].detach(), outputs[OUTPUT_TEACHER_TRACK_EMBEDDING_REF]

        loss_dict = {}

        batch_size = bbox_preds.shape[0]
        track_similarity_losses = []

        for j in range(batch_size):
            gt_label = gt_labels[j]
            gt_bbox_2d = gt_bboxes_2d[j].floor()
            gt_reid = gt_reids[j]
            bbox_pred = bbox_preds[j, :, :4]
            embedding_pred = embedding_preds[j]

            gt_label_ref = gt_labels_ref[j]
            gt_bbox_2d_ref = gt_bboxes_2d_ref[j].floor()
            gt_reid_ref = gt_reids_ref[j]
            bbox_pred_ref = bbox_preds_ref[j, :, :4]
            embedding_pred_ref = embedding_preds_ref[j]

            bbox_pred, gt_reid, embedding_pred = self.get_bbox_target(gt_label, gt_bbox_2d, gt_reid, bbox_pred, embedding_pred, True)            
            bbox_pred_ps, gt_reid_ps, embedding_pred_ps = self.get_bbox_target(gt_label_ref, gt_bbox_2d_ref, gt_reid_ref, bbox_pred_ref, embedding_pred_ref, True)

            if gt_reid is not None and gt_reid.shape[0] and gt_reid_ps is not None and gt_reid_ps.shape[0] > 0:
                    bbox_pred_ng, gt_reid_ng, embedding_pred_ng = self.get_bbox_target(gt_label_ref, gt_bbox_2d_ref, gt_reid_ref, bbox_pred_ref, embedding_pred_ref, False)
                    gt_reid_ref = torch.cat((gt_reid_ps, gt_reid_ng), 0)
                    embedding_pred_ref = torch.cat((embedding_pred_ps, embedding_pred_ng), 0)
                    row = gt_reid.shape[0]
                    col = gt_reid_ref.shape[0]
                    
                    match = gt_reid.view(row, 1).repeat(1, col) == gt_reid_ref.view(1, col).repeat(row, 1)
                    targets = torch.ones_like(match).cuda()*(-1)
                    targets[match] = 1
                    
                    embedding_pred = F.normalize(embedding_pred)
                    embedding_pred_ref = F.normalize(embedding_pred_ref)
                    similarity = torch.mm(embedding_pred, embedding_pred_ref.t())
                    similarity = torch.clamp(similarity, min=-1+1e-4, max=1 - 1e-4)
                    
                    weights = (match.sum(dim=1) > 0).float() + 1e-4
                    similarity_loss = self.loss_track(similarity, targets, weights, avg_factor=weights.sum())

                    track_similarity_losses.append(similarity_loss)
        if len(track_similarity_losses) == 0:
            loss_dict["loss_track_similarity"] = torch.tensor(0.0).cuda()
        else:
            loss_dict["loss_track_similarity"] = torch.stack(track_similarity_losses).mean(dim=0, keepdim=True)

        return loss_dict
   
    def gather_feat(self, feat, ind, mask=None):
        dim = feat.size(1)
        ind = ind.unsqueeze(1).expand(ind.size(0), dim)
        feat = feat.gather(0, ind)
        return feat

    def get_bbox_target(self, gt_label, gt_bbox_2d, gt_reid, bbox_pred, embedding_pred, is_positive, ng_id=-1):
        idx = (gt_label < self.num_cls) & (gt_reid >= 0)
        bbox_2d = gt_bbox_2d[idx]

        if bbox_2d.shape[0] == 0:
            return None, None, None

        label = gt_label[idx]
        reid = gt_reid[idx]
        embedding_dim, height, width = embedding_pred.shape
        embedding_pred = embedding_pred.permute(1,2,0).contiguous().view(height*width, embedding_dim)
        
        if is_positive:
            ious = self.bbox_overlaps(self.xywh2xyxy(bbox_pred), bbox_2d)
            max_iou, max_id = ious.max(dim=1)
            idx = max_iou >= self.positive_iou_threshold
            max_id = max_id[idx]
            bbox_pred = bbox_pred[idx]
            reid_pred = reid[max_id]
        else:
            ious = self.bbox_overlaps(self.xywh2xyxy(bbox_pred), bbox_2d)
            max_iou, max_id = ious.max(dim=1)
            idx = max_iou <= self.negative_iou_threshold
            bbox_pred = bbox_pred[idx]
            max_iou = max_iou[idx]

            topk_iou, topk_index = torch.topk(max_iou, 2048)
            bbox_pred = bbox_pred[topk_index.squeeze()]
            
            # Todo random choice
            indices = torch.randperm(topk_iou.shape[0])[:128]
            bbox_pred = bbox_pred[indices]

            # Todo size limit
            indices = (bbox_pred[:, 0] >= 0) & (bbox_pred[:, 0] < 768) & (bbox_pred[:, 1] >= 0) & (bbox_pred[:, 1] < 384)
            bbox_pred = bbox_pred[indices]
            reid_pred = (torch.ones(bbox_pred.shape[0]).cuda()*ng_id).long()

        # Todo Align
        bbox_pred_ct_int_x = (bbox_pred[:, 0]/self.down_ratio).int()
        bbox_pred_ct_int_y = (bbox_pred[:, 1]/self.down_ratio).int()
        bbox_pred_ct_int_x_ = bbox_pred_ct_int_x+1
        bbox_pred_ct_int_x_[bbox_pred_ct_int_x_== width] -= 1
        bbox_pred_ct_int_y_ = bbox_pred_ct_int_y+1
        bbox_pred_ct_int_y_[bbox_pred_ct_int_y_== height] -= 1 
        
        ct_res_x = (bbox_pred[:, 0]/self.down_ratio - bbox_pred_ct_int_x).view(-1, 1).contiguous()
        ct_res_y = (bbox_pred[:, 1]/self.down_ratio - bbox_pred_ct_int_y).view(-1, 1).contiguous()

        ct_index_0 = (bbox_pred_ct_int_x + width * bbox_pred_ct_int_y).long()
        ct_index_1 = (bbox_pred_ct_int_x + width * bbox_pred_ct_int_y_).long()
        ct_index_2 = (bbox_pred_ct_int_x_ + width * bbox_pred_ct_int_y).long()
        ct_index_3 = (bbox_pred_ct_int_x_ + width * bbox_pred_ct_int_y_).long()

        embedding_pred = (1-ct_res_x)*(1-ct_res_y) * self.gather_feat(embedding_pred, ct_index_0) \
                            + (1-ct_res_x) * ct_res_y * self.gather_feat(embedding_pred, ct_index_1) \
                            + ct_res_x * (1-ct_res_y) * self.gather_feat(embedding_pred, ct_index_2) \
                            + ct_res_x * ct_res_y * self.gather_feat(embedding_pred, ct_index_3) 

        # Todo
        # bbox_pred_ct_int = (bbox_pred[:, :2]/self.down_ratio).long()
        # bbox_pred_ct_index = bbox_pred_ct_int[:, 0] + width*bbox_pred_ct_int[:, 1]
        # embedding_pred = torch.index_select(embedding_pred, 0, bbox_pred_ct_index)

        return bbox_pred, reid_pred, embedding_pred

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def bbox_overlaps(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate overlap between two set of bboxes.

        If ``is_aligned`` is ``False``, then calculate the ious between each bbox
        of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
        bboxes1 and bboxes2.

        Args:
            bboxes1 (Tensor): shape (m, 4)
            bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
                must be equal.
            mode (str): "iou" (intersection over union) or iof (intersection over
                foreground).

        Returns:
            ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
        """

        assert mode in ['iou', 'iof']

        rows = bboxes1.size(0)
        cols = bboxes2.size(0)
        if is_aligned:
            assert rows == cols

        if rows * cols == 0:
            return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

        if is_aligned:
            lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
            rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

            wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
            overlap = wh[:, 0] * wh[:, 1]
            area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
                bboxes1[:, 3] - bboxes1[:, 1] + 1)

            if mode == 'iou':
                area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                    bboxes2[:, 3] - bboxes2[:, 1] + 1)
                ious = overlap / (area1 + area2 - overlap)
            else:
                ious = overlap / area1
        else:
            lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
            rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

            wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
            overlap = wh[:, :, 0] * wh[:, :, 1]
            area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
                bboxes1[:, 3] - bboxes1[:, 1] + 1)

            if mode == 'iou':
                area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                    bboxes2[:, 3] - bboxes2[:, 1] + 1)
                ious = overlap / (area1[:, None] + area2 - overlap)
            else:
                ious = overlap / (area1[:, None])

        return ious