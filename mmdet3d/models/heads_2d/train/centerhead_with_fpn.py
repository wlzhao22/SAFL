from mmcv.cnn.bricks.norm import build_norm_layer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmdet3d.models.utils.utils_2d.efficientdet_utils import tranpose_and_gather_feat, ConvBlock
from mmdet3d.models.utils.utils_2d.gaussian_target import draw_umich_gaussian, gaussian_radius
from mmdet3d.models.utils.utils_2d.key_config import *
from mmdet3d.models.utils.utils_2d.instances import *
from mmdet3d.models.utils.utils_2d.boxes import *
from mmdet3d.models.builder import HEADS
from mmdet3d.models.losses.losses_2d.heatmap_focal_loss import *
from mmdet3d.models.losses.losses_2d.centernet_iou_loss import *
from torchvision.ops.boxes import batched_nms, nms
import torch.distributed as dist

# HEADS._module_dict.pop('CenterNetHead')

import torch
from torch import nn
INF = 100000000 

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

@HEADS.register_module()
class CenterNetFPNHead(nn.Module):
    def __init__(self, 
            num_classes,
            in_channel, 
            input_shape, 
            image_size,
            out_kernel=3,
            num_convs=4,
            # strides = [8, 16, 32, 64, 128],
            strides = [8, 16, 32],
            score_thresh = 0.05,
            nms_thresh_train = 0.9,
            nms_thresh_test = 0.9,
            pre_nms_topk_train = 4000,
            post_nms_topk_train = 2000,
            pre_nms_topk_test = 1000,
            post_nms_topk_test = 100,
            iou_loss_type = 'giou',
            hm_focal_alpha = 0.25,
            hm_focal_beta = 4,
            loss_gamma = 2,
            sigmoid_clamp = 1e-4,
            ignore_high_fp = 0.85,
            pos_weight = 1,
            neg_weight = 1,
            more_pos = False, # Todo
            not_norm_reg = True,
            delta = (1 - 0.8) / (1 + 0.8),
            min_radius = 4,
            # sizes_of_interest = [[0, 80], [64, 160], [128, 320], [256, 640], [512, 10000000]],
            sizes_of_interest = [[0, 64], [48, 192], [128, 1000000]],
            with_agn_hm = True,
            center_nms = False,
            not_nms = False,
            only_proposal = False,
            as_proposal = False,
            reg_weight = 2,
            is_requires_grad = True,
            norm_cfg=dict(type='GN', num_groups=32),
        ):
            
        super().__init__()
        self.num_classes = num_classes
        self.out_kernel = out_kernel
        self.num_convs = num_convs
        self.strides = strides
        self.score_thresh = score_thresh
        self.nms_thresh_train = nms_thresh_train
        self.nms_thresh_test = nms_thresh_test
        self.image_size = tuple(image_size)
        self.pre_nms_topk_train = pre_nms_topk_train
        self.post_nms_topk_train = post_nms_topk_train
        self.pre_nms_topk_test = pre_nms_topk_test
        self.post_nms_topk_test = post_nms_topk_test
        self.iou_loss = IOULoss(iou_loss_type)
        self.hm_focal_alpha = hm_focal_alpha
        self.hm_focal_beta = hm_focal_beta
        self.loss_gamma = loss_gamma
        self.sigmoid_clamp = sigmoid_clamp
        self.ignore_high_fp = ignore_high_fp
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.more_pos = more_pos
        self.not_norm_reg = not_norm_reg
        self.delta = delta
        self.min_radius = min_radius
        self.sizes_of_interest = sizes_of_interest
        self.with_agn_hm = with_agn_hm
        self.center_nms = center_nms
        self.only_proposal = only_proposal
        self.as_proposal = as_proposal
        self.reg_weight = reg_weight
        self.not_nms = not_nms

        head_configs = {"cls": 4 if not self.only_proposal else 0,
                        "bbox": 4}

        channels = {
            'cls': in_channel,
            'bbox': in_channel,
        }
        for head in head_configs:
            tower = []
            num_convs = head_configs[head]
            for i in range(num_convs):
                tower.append(nn.Conv2d(
                    in_channel,
                    in_channel, 
                    kernel_size=3, stride=1,
                    padding=1, bias=True
                ))
                tower.append(build_norm_layer(norm_cfg, in_channel)[1])
                tower.append(nn.ReLU(inplace=True))
            self.add_module('{}_tower'.format(head), nn.Sequential(*tower))

        self.bbox_pred = nn.Conv2d(in_channel, 4, kernel_size=self.out_kernel,stride=1, padding=self.out_kernel // 2)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in input_shape])
        
        for modules in [self.cls_tower, self.bbox_tower, self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.bbox_pred.bias, 8.)

        if self.with_agn_hm:
            self.agn_hm = nn.Conv2d(in_channel, 1, kernel_size=self.out_kernel, stride=1, padding=self.out_kernel // 2)
            torch.nn.init.constant_(self.agn_hm.bias, bias_value)
            torch.nn.init.normal_(self.agn_hm.weight, std=0.01)

        if not self.only_proposal:
            cls_kernel_size = self.out_kernel
            self.cls_logits = nn.Conv2d(in_channel, self.num_classes,
                kernel_size=cls_kernel_size, 
                stride=1,
                padding=cls_kernel_size // 2,
            )
            torch.nn.init.constant_(self.cls_logits.bias, bias_value)
            torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        
        if not is_requires_grad:
            for p in self.parameters():
                p.requires_grad = False
            # self.eval()

    def load_pretrain(self, pretrained_model, pre_name):
        pretrained_dict = dict(torch.load(pretrained_model)['state_dict'])
        model_dict = self.state_dict()
        pretrained_dict = {k: pretrained_dict[pre_name+'.'+ k] for k, v in dict(model_dict).items() if pre_name+'.'+k in pretrained_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x, *_):
        clss = []
        bbox_reg = []
        agn_hms = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            if not self.only_proposal:
                clss.append(self.cls_logits(cls_tower))
            else:
                clss.append(None)
            
            if self.with_agn_hm:
                agn_hms.append(self.agn_hm(bbox_tower))
            else:
                agn_hms.append(None)
            reg = self.bbox_pred(bbox_tower)
            reg = self.scales[l](reg)
            bbox_reg.append(F.relu(reg))

        grids = self._compute_grids(x)
        output = {}

        if not self.training:
            # proposals, _ = self.inference(clss, bbox_reg, agn_hms, grids)
            logits_pred = [x.sigmoid() if x is not None else None for x in clss]
            agn_hms = [x.sigmoid() if x is not None else None for x in agn_hms]
            if self.only_proposal:
                proposals = self._predict_instances(grids, agn_hms, bbox_reg, [None for _ in agn_hms])
                for p in range(len(proposals)):
                    proposals[p].proposal_boxes = proposals[p].get('pred_boxes')
                    proposals[p].objectness_logits = proposals[p].get('scores')
                    proposals[p].remove('pred_boxes')
                    proposals[p].remove('scores')
                    proposals[p].remove('pred_classes')
                output['proposals'] = proposals
            else:
                proposals = self._predict_instances(grids, logits_pred, bbox_reg, agn_hms)
                proposal_boxes = [p.get('pred_boxes').tensor for p in proposals]
                objectness_logits = [p.get('scores') for p in proposals]
                objectness_classes = [p.get('pred_classes') for p in proposals]
                output = {'bbox': proposal_boxes, 'score': objectness_logits, 'label': objectness_classes}
        else:
            output = {'bbox': bbox_reg, 'agn': agn_hms, 'clss': clss}
            if self.only_proposal:
                agn_hms = [x.sigmoid() for x in agn_hms]
                proposals = self._predict_instances(grids, agn_hms, bbox_reg, [None for _ in agn_hms])
                
                for p in range(len(proposals)):
                    proposals[p].proposal_boxes = proposals[p].get('pred_boxes')
                    proposals[p].objectness_logits = proposals[p].get('scores')
                    proposals[p].remove('pred_boxes')
                    proposals[p].remove('scores')
                    proposals[p].remove('pred_classes')
                output['proposals'] = proposals
        return output
    
    def forward_train(self, inputs, outputs):
        features = outputs['features']
        res = self.forward(features)
        outputs['proposals'] = res['proposals']
        clss_per_level, reg_pred_per_level, agn_hm_pred_per_level = res['clss'], res['bbox'], res['agn']
        grids = self._compute_grids(features)
        shapes_per_level = grids[0].new_tensor([(x.shape[2], x.shape[3]) for x in reg_pred_per_level])
        
        gt_bboxes_2d, gt_labels = inputs['gt_bboxes_2d'], inputs['gt_labels']
        gt_instances = []
        for B in range(len(gt_bboxes_2d)):
            gt_instance = Instances(self.image_size)
            gt_instance.gt_boxes = Boxes(gt_bboxes_2d[B])
            gt_instance.gt_classes = gt_labels[B]    
            gt_instances.append(gt_instance)
        inputs['targets'] = gt_instances

        pos_inds, labels, reg_targets, flattened_hms = self._get_ground_truth(grids, shapes_per_level, gt_instances)        
        logits_pred, reg_pred, agn_hm_pred = self._flatten_outputs(clss_per_level, reg_pred_per_level, agn_hm_pred_per_level)
        losses = self.losses(pos_inds, labels, reg_targets, flattened_hms, logits_pred, reg_pred, agn_hm_pred)
        
        return losses

    def losses(
        self, pos_inds, labels, reg_targets, flattened_hms,
        logits_pred, reg_pred, agn_hm_pred):
        '''
        Inputs:
            pos_inds: N
            labels: N
            reg_targets: M x 4
            flattened_hms: M x C
            logits_pred: M x C
            reg_pred: M x 4
            agn_hm_pred: M x 1 or None
            N: number of positive locations in all images
            M: number of pixels from all FPN levels
            C: number of classes
        '''
        assert (torch.isfinite(reg_pred).all().item())
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(
            pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        losses = {}
        if not self.only_proposal:
            pos_loss, neg_loss = heatmap_focal_loss_jit(
                logits_pred, flattened_hms, pos_inds, labels,
                alpha=self.hm_focal_alpha, 
                beta=self.hm_focal_beta, 
                gamma=self.loss_gamma, 
                reduction='mean',
                sigmoid_clamp=self.sigmoid_clamp,
                ignore_high_fp=self.ignore_high_fp,
            )
            pos_loss = self.pos_weight * pos_loss / num_pos_avg
            neg_loss = self.neg_weight * neg_loss / num_pos_avg
            # pos_loss = self.pos_weight * pos_loss
            # neg_loss = self.neg_weight * neg_loss
            losses['loss_centernet_pos'] = pos_loss
            losses['loss_centernet_neg'] = neg_loss
        
        reg_inds = torch.nonzero(reg_targets.max(dim=1)[0] >= 0).squeeze(1)
        reg_pred = reg_pred[reg_inds]
        reg_targets_pos = reg_targets[reg_inds]
        reg_weight_map = flattened_hms.max(dim=1)[0]
        reg_weight_map = reg_weight_map[reg_inds]
        reg_weight_map = reg_weight_map * 0 + 1 if self.not_norm_reg else reg_weight_map
        reg_norm = max(reduce_sum(reg_weight_map.sum()).item() / num_gpus, 1)
        reg_loss = self.reg_weight * self.iou_loss(reg_pred, reg_targets_pos, reg_weight_map,reduction='sum') / reg_norm
        losses['loss_centernet_loc'] = reg_loss

        if self.with_agn_hm:
            cat_agn_heatmap = flattened_hms.max(dim=1)[0] # M
            agn_pos_loss, agn_neg_loss = binary_heatmap_focal_loss_jit(
                agn_hm_pred, cat_agn_heatmap, pos_inds,
                alpha=self.hm_focal_alpha, 
                beta=self.hm_focal_beta, 
                gamma=self.loss_gamma,
                sigmoid_clamp=self.sigmoid_clamp,
                ignore_high_fp=self.ignore_high_fp,
            )
            agn_pos_loss = self.pos_weight * agn_pos_loss / num_pos_avg
            agn_neg_loss = self.neg_weight * agn_neg_loss / num_pos_avg
            # agn_pos_loss = self.pos_weight * agn_pos_loss
            # agn_neg_loss = self.neg_weight * agn_neg_loss
            losses['loss_centernet_agn_pos'] = agn_pos_loss
            losses['loss_centernet_agn_neg'] = agn_neg_loss

        return losses

    def _compute_grids(self, features):
        grids = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            shifts_x = torch.arange(
                0, w * self.strides[level], 
                step=self.strides[level],
                dtype=torch.float32, device=feature.device)
            shifts_y = torch.arange(
                0, h * self.strides[level], 
                step=self.strides[level],
                dtype=torch.float32, device=feature.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            grids_per_level = torch.stack((shift_x, shift_y), dim=1) + \
                self.strides[level] // 2
            grids.append(grids_per_level)
        return grids
    
    def _predict_instances(self, grids, logits_pred, reg_pred, agn_hm_pred):
        sampled_boxes = []
        for l in range(len(grids)):
            sampled_boxes.append(self._predict_single_level(grids[l], logits_pred[l], reg_pred[l] * self.strides[l], agn_hm_pred[l], l))
        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self._nms_and_topK(boxlists, nms=not self.not_nms)
        return boxlists

    def _predict_single_level(self, grids, heatmap, reg_pred, agn_hm, level):
        N, C, H, W = heatmap.shape
        # put in the same format as grids
        if self.center_nms:
            heatmap_nms = nn.functional.max_pool2d(
                heatmap, (3, 3), stride=1, padding=1)
            heatmap = heatmap * (heatmap_nms == heatmap).float()
        heatmap = heatmap.permute(0, 2, 3, 1) # N x H x W x C
        heatmap = heatmap.reshape(N, -1, C) # N x HW x C
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1) # N x H x W x 4 
        box_regression = box_regression.reshape(N, -1, 4)

        candidate_inds = heatmap > self.score_thresh # 0.05
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1) # N
        pre_nms_topk = self.pre_nms_topk_train if self.training else self.pre_nms_topk_test
        pre_nms_top_n = pre_nms_top_n.clamp(max=pre_nms_topk) # N

        if agn_hm is not None:
            agn_hm = agn_hm.view(N, 1, H, W).permute(0, 2, 3, 1)
            agn_hm = agn_hm.reshape(N, -1)
            heatmap = heatmap * agn_hm[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = heatmap[i] # HW x C
            per_candidate_inds = candidate_inds[i] # n
            per_box_cls = per_box_cls[per_candidate_inds] # n

            per_candidate_nonzeros = per_candidate_inds.nonzero() # n
            per_box_loc = per_candidate_nonzeros[:, 0] # n
            per_class = per_candidate_nonzeros[:, 1] # n

            per_box_regression = box_regression[i] # HW x 4
            per_box_regression = per_box_regression[per_box_loc] # n x 4
            per_grids = grids[per_box_loc] # n x 2

            per_pre_nms_top_n = pre_nms_top_n[i] # 1

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_grids = per_grids[top_k_indices]
            
            detections = torch.stack([
                per_grids[:, 0] - per_box_regression[:, 0],
                per_grids[:, 1] - per_box_regression[:, 1],
                per_grids[:, 0] + per_box_regression[:, 2],
                per_grids[:, 1] + per_box_regression[:, 3],
            ], dim=1) # n x 4

            # avoid invalid boxes in RoI heads
            detections[:, 2] = torch.max(detections[:, 2], detections[:, 0] + 0.01)
            detections[:, 3] = torch.max(detections[:, 3], detections[:, 1] + 0.01)
            boxlist = Instances(self.image_size)
            boxlist.scores = torch.sqrt(per_box_cls) if self.with_agn_hm else per_box_cls # n
            # import pdb; pdb.set_trace()
            boxlist.pred_boxes = Boxes(detections)
            boxlist.pred_classes = per_class
            results.append(boxlist)
        return results

    def _nms_and_topK(self, boxlists, nms=True):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            nms_thresh = self.nms_thresh_train if self.training else self.nms_thresh_test
            result = self._ml_nms(boxlists[i], nms_thresh) if nms else boxlists[i]
            num_dets = len(result)
            post_nms_topk = self.post_nms_topk_train if self.training else self.post_nms_topk_test
            if num_dets > post_nms_topk:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    num_dets - post_nms_topk + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results
    
    def _ml_nms(self, boxlist, nms_thresh, max_proposals=-1, score_field="scores", label_field="labels"):
        """
        Performs non-maximum suppression on a boxlist, with scores specified
        in a boxlist field via score_field.
        Arguments:
            boxlist(BoxList)
            nms_thresh (float)
            max_proposals (int): if > 0, then only the top max_proposals are kept
                after non-maximum suppression
            score_field (str)
        """
        if nms_thresh <= 0:
            return boxlist
        if boxlist.has('pred_boxes'):
            boxes = boxlist.pred_boxes.tensor
            labels = boxlist.pred_classes
        else:
            boxes = boxlist.proposal_boxes.tensor
            labels = boxlist.proposal_boxes.tensor.new_zeros(
                len(boxlist.proposal_boxes.tensor))
        scores = boxlist.scores
        
        # keep = batched_nms(boxes, scores, labels, nms_thresh)
        keep = nms(boxes, scores, nms_thresh)
        if max_proposals > 0:
            keep = keep[: max_proposals]
        boxlist = boxlist[keep]
        return boxlist

    def _get_center3x3(self, locations, centers, strides):
        '''
        Inputs:
            locations: M x 2
            centers: N x 2
            strides: M
        '''
        M, N = locations.shape[0], centers.shape[0]
        locations_expanded = locations.view(M, 1, 2).expand(M, N, 2) # M x N x 2
        centers_expanded = centers.view(1, N, 2).expand(M, N, 2) # M x N x 2
        strides_expanded = strides.view(M, 1, 1).expand(M, N, 2) # M x N
        centers_discret = ((centers_expanded / strides_expanded).int() * \
            strides_expanded).float() + strides_expanded / 2 # M x N x 2
        dist_x = (locations_expanded[:, :, 0] - centers_discret[:, :, 0]).abs()
        dist_y = (locations_expanded[:, :, 1] - centers_discret[:, :, 1]).abs()
        return (dist_x <= strides_expanded[:, :, 0]) & \
            (dist_y <= strides_expanded[:, :, 0])

    def _assign_reg_fpn(self, reg_targets_per_im, size_ranges):
        '''
        TODO (Xingyi): merge it with assign_fpn_level
        Inputs:
            reg_targets_per_im: M x N x 4
            size_ranges: M x 2
        '''
        crit = ((reg_targets_per_im[:, :, :2] + \
            reg_targets_per_im[:, :, 2:])**2).sum(dim=2) ** 0.5 / 2 # M x N
        is_cared_in_the_level = (crit >= size_ranges[:, [0]]) & \
            (crit <= size_ranges[:, [1]])
        return is_cared_in_the_level

    def _get_ground_truth(self, grids, shapes_per_level, gt_instances):
        '''
        Input:
            grids: list of tensors [(hl x wl, 2)]_l
            shapes_per_level: list of tuples L x 2:
            gt_instances: gt instances
        Retuen:
            pos_inds: N
            labels: N
            reg_targets: M x 4
            flattened_hms: M x C or M x 1
            N: number of objects in all images
            M: number of pixels from all FPN levels
        '''

        # get positive pixel index
        pos_inds, labels = self._get_label_inds(gt_instances, shapes_per_level)
        heatmap_channels = self.num_classes
        L = len(grids)
        num_loc_list = [len(loc) for loc in grids]
        strides = torch.cat([
            shapes_per_level.new_ones(num_loc_list[l]) * self.strides[l] \
            for l in range(L)]).float() # M
        reg_size_ranges = torch.cat([
            shapes_per_level.new_tensor(self.sizes_of_interest[l]).float().view(
            1, 2).expand(num_loc_list[l], 2) for l in range(L)]) # M x 2
        grids = torch.cat(grids, dim=0) # M x 2
        M = grids.shape[0]

        reg_targets = []
        flattened_hms = []
        for i in range(len(gt_instances)): # images
            boxes = gt_instances[i].gt_boxes.tensor # N x 4
            area = gt_instances[i].gt_boxes.area() # N
            gt_classes = gt_instances[i].gt_classes # N in [0, self.num_classes]

            N = boxes.shape[0]
            if N == 0:
                reg_targets.append(grids.new_zeros((M, 4)) - INF)
                flattened_hms.append(
                    grids.new_zeros((
                        M, 1 if self.only_proposal else heatmap_channels)))
                continue
            
            l = grids[:, 0].view(M, 1) - boxes[:, 0].view(1, N) # M x N
            t = grids[:, 1].view(M, 1) - boxes[:, 1].view(1, N) # M x N
            r = boxes[:, 2].view(1, N) - grids[:, 0].view(M, 1) # M x N
            b = boxes[:, 3].view(1, N) - grids[:, 1].view(M, 1) # M x N
            reg_target = torch.stack([l, t, r, b], dim=2) # M x N x 4

            centers = ((boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2) # N x 2
            centers_expanded = centers.view(1, N, 2).expand(M, N, 2) # M x N x 2
            strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)
            centers_discret = ((centers_expanded / strides_expanded).int() * \
                strides_expanded).float() + strides_expanded / 2 # M x N x 2
            
            is_peak = (((grids.view(M, 1, 2).expand(M, N, 2) - \
                centers_discret) ** 2).sum(dim=2) == 0) # M x N
            is_in_boxes = reg_target.min(dim=2)[0] > 0 # M x N
            is_center3x3 = self._get_center3x3(
                grids, centers, strides) & is_in_boxes # M x N
            is_cared_in_the_level = self._assign_reg_fpn(
                reg_target, reg_size_ranges) # M x N
            reg_mask = is_center3x3 & is_cared_in_the_level # M x N

            dist2 = ((grids.view(M, 1, 2).expand(M, N, 2) - centers_expanded) ** 2).sum(dim=2) # M x N
            dist2[is_peak] = 0
            radius2 = self.delta ** 2 * 2 * area # N
            radius2 = torch.clamp(
                radius2, min=self.min_radius ** 2)
            weighted_dist2 = dist2 / radius2.view(1, N).expand(M, N) # M x N            
            reg_target = self._get_reg_targets(
                reg_target, weighted_dist2.clone(), reg_mask, area) # M x 4

            if self.only_proposal:
                flattened_hm = self._create_agn_heatmaps_from_dist(
                    weighted_dist2.clone()) # M x 1
            else:
                flattened_hm = self._create_heatmaps_from_dist(
                    weighted_dist2.clone(), gt_classes, 
                    channels=heatmap_channels) # M x C

            reg_targets.append(reg_target)
            flattened_hms.append(flattened_hm)
        
        # transpose im first training_targets to level first ones
        reg_targets = self._transpose(reg_targets, num_loc_list)
        flattened_hms = self._transpose(flattened_hms, num_loc_list)
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.strides[l])
        reg_targets = cat([x for x in reg_targets], dim=0) # MB x 4
        flattened_hms = cat([x for x in flattened_hms], dim=0) # MB x C
        
        return pos_inds, labels, reg_targets, flattened_hms
    
    def _get_label_inds(self, gt_instances, shapes_per_level):
        '''
        Inputs:
            gt_instances: [n_i], sum n_i = N
            shapes_per_level: L x 2 [(h_l, w_l)]_L
        Returns:
            pos_inds: N'
            labels: N'
        '''
        pos_inds = []
        labels = []
        L = len(self.strides)
        B = len(gt_instances)
        shapes_per_level = shapes_per_level.long()
        loc_per_level = (shapes_per_level[:, 0] * shapes_per_level[:, 1]).long() # L
        level_bases = []
        s = 0
        for l in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[l]
        level_bases = shapes_per_level.new_tensor(level_bases).long() # L
        strides_default = shapes_per_level.new_tensor(self.strides).float() # L
        for im_i in range(B):
            targets_per_im = gt_instances[im_i]
            bboxes = targets_per_im.gt_boxes.tensor # n x 4
            n = bboxes.shape[0]
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2) # n x 2
            centers = centers.view(n, 1, 2).expand(n, L, 2)
            strides = strides_default.view(1, L, 1).expand(n, L, 2)
            centers_inds = (centers / strides).long() # n x L x 2
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            pos_ind = level_bases.view(1, L).expand(n, L) + \
                       im_i * loc_per_level.view(1, L).expand(n, L) + \
                       centers_inds[:, :, 1] * Ws + \
                       centers_inds[:, :, 0] # n x L
            is_cared_in_the_level = self._assign_fpn_level(bboxes)
            pos_ind = pos_ind[is_cared_in_the_level].view(-1)
            label = targets_per_im.gt_classes.view(n, 1).expand(n, L)[is_cared_in_the_level].view(-1)

            pos_inds.append(pos_ind) # n'
            labels.append(label.long()) # n'
        pos_inds = torch.cat(pos_inds, dim=0).long()
        labels = torch.cat(labels, dim=0)
        return pos_inds, labels # N, N
    
    def _flatten_outputs(self, clss, reg_pred, agn_hm_pred):
        # Reshape: (N, F, Hl, Wl) -> (N, Hl, Wl, F) -> (sum_l N*Hl*Wl, F)
        clss = cat([x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]) for x in clss], dim=0) if clss[0] is not None else None
        reg_pred = cat([x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred], dim=0)            
        agn_hm_pred = cat([x.permute(0, 2, 3, 1).reshape(-1) for x in agn_hm_pred], dim=0) if self.with_agn_hm else None
        return clss, reg_pred, agn_hm_pred

    def _get_reg_targets(self, reg_targets, dist, mask, area):
        '''
          reg_targets (M x N x 4): long tensor
          dist (M x N)
          is_*: M x N
        '''
        dist[mask == 0] = INF * 1.0
        min_dist, min_inds = dist.min(dim=1) # M
        reg_targets_per_im = reg_targets[
            range(len(reg_targets)), min_inds] # M x N x 4 --> M x 4
        reg_targets_per_im[min_dist == INF] = - INF
        return reg_targets_per_im        
    
    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to 
            level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0)

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0))
        return targets_level_first

    def _create_heatmaps_from_dist(self, dist, labels, channels):
        '''
        dist: M x N
        labels: N
        return:
          heatmaps: M x C
        '''
        heatmaps = dist.new_zeros((dist.shape[0], channels))
        for c in range(channels):
            inds = (labels == c) # N
            if inds.int().sum() == 0:
                continue
            heatmaps[:, c] = torch.exp(-dist[:, inds].min(dim=1)[0])
            zeros = heatmaps[:, c] < 1e-4
            heatmaps[zeros, c] = 0
        return heatmaps

    def _create_agn_heatmaps_from_dist(self, dist):
        '''
        TODO (Xingyi): merge it with _create_heatmaps_from_dist
        dist: M x N
        return:
          heatmaps: M x 1
        '''
        heatmaps = dist.new_zeros((dist.shape[0], 1))
        heatmaps[:, 0] = torch.exp(-dist.min(dim=1)[0])
        zeros = heatmaps < 1e-4
        heatmaps[zeros] = 0
        return heatmaps

    def _assign_fpn_level(self, boxes):
        '''
        Inputs:
            boxes: n x 4
            size_ranges: L x 2
        Return:
            is_cared_in_the_level: n x L
        '''
        size_ranges = boxes.new_tensor(
            self.sizes_of_interest).view(len(self.sizes_of_interest), 2) # L x 2
        crit = ((boxes[:, 2:] - boxes[:, :2]) **2).sum(dim=1) ** 0.5 / 2 # n
        n, L = crit.shape[0], size_ranges.shape[0]
        crit = crit.view(n, 1).expand(n, L)
        size_ranges_expand = size_ranges.view(1, L, 2).expand(n, L, 2)
        is_cared_in_the_level = (crit >= size_ranges_expand[:, :, 0]) & \
            (crit <= size_ranges_expand[:, :, 1])
        return is_cared_in_the_level
    