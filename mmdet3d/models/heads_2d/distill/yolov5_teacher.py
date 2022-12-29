from mmdet3d.models.utils.utils_2d.key_config import *
from mmdet.core import multi_apply
from mmdet3d.models.utils.utils_2d.yolov5_utils import *
from mmdet3d.models.builder import HEADS
import torch.nn.functional as F
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super(ConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.cov3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cov1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cov3x3(x)
        x = self.cov1x1(x)

        return x

@HEADS.register_module()
class YOLOv5HeadTea(nn.Module):

    def __init__(self, img_shape, in_channels, num_class, anchors,bins=8,  with_ref=True, overlap = 25/180.0 * np.pi,with_ddd=False, bbox_size_anchor=None, position=None, label_smooth=True, deta=0.01, anchor_t=4, anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], downsample_ratios=[32, 16, 8], conf_balances=[0.4,1,4], nms_thr=0.5, scores_thr=0.1, merge=False):
        super(YOLOv5HeadTea, self).__init__()
        # x, y, w, h, conf, short_x, lf, rf, rr, lr, h, w, l, right, rightFront, front, leftFront, left, leftRear, rear, rightRear, bin, res_sin, res_cos
        self.with_ddd = with_ddd
        if self.with_ddd:
            self.base_num = 4 + 1 + 5 + 3 + 8 + bins*3            
        else:
            self.base_num = 5
        self.num_classes = num_class  # number of classes
        self.no = num_class + self.base_num # number of outputs per anchor
        self.nl = 3  # number of detection layers
        self.na = len(anchor_masks)  # number of anchors
        self.anchors = anchors
        self.with_ref = with_ref

        self.short_anchors = []
        for anchor in self.anchors:
            self.short_anchors.append([anchor[0], anchor[1], anchor[1], anchor[1], anchor[1]])

        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.class_loss = Class_Loss(loss_weight=0.5,pos_weight=1.0)
        self.confidence_loss = Conf_Loss(loss_weight=1,pos_weight=1.0)
        self.bbox_loss = IOU_Loss(iou_type='GIOU',loss_weight=0.5)
        self.conf_balances = conf_balances
        self.scores_thr = scores_thr
        self.merge = merge
        self.nms_thr = nms_thr
        self.anchor_t = anchor_t
        self.label_smooth = label_smooth
        self.deta = deta
        self.img_shape = img_shape
        self.detect_head = nn.ModuleList(ConvBlock(x, self.no * self.na) for x in in_channels)  # output conv
        # self.cSE_head = nn.ModuleList(cSE(x) for x in in_channels)  # output conv
        self.position = position

        self.bbox_size_anchor = bbox_size_anchor
        self.bins = bins
        self.intervalAngle = 2 * np.pi / self.bins
        self.overlap = overlap
        self.centerAngle = np.array([(i + 1-self.bins/2) * self.intervalAngle for i in range(self.bins)])
        self.max_num_box = 256

    def forward_single(self, feat):
        detection_list = []
        detection_bbox_list = []

        for i in range(self.nl):
            if not self.training or self.position is None:
                # det = self.detect_head[i](self.cSE_head[i](feat[self.nl-i-1]))
                 det = self.detect_head[i](feat[self.nl-i-1])
            else:
                # det = self.detect_head[i](self.cSE_head[i](feat[self.nl-i-1][self.index_bool]))  # conv
                det = self.detect_head[i](feat[self.nl-i-1][self.index_bool])  # conv

            bs, _, ny, nx = det.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            det = det.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # if not self.training:
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid_xy = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(det.device)
            y = det
            if self.with_ddd:
                det_1 = y[..., 0:18 + self.bins].sigmoid()
                det_2 = y[..., 18 + self.bins:18 + 3 * self.bins].sigmoid()
                det_3 = y[..., 18 + 3 * self.bins:].sigmoid()

                det_11 = (det_1[..., 0:2] * 2. - 0.5 + grid_xy) * self.downsample_ratios[i]  # xy 映射回原图位置
                det_12 = (det_1[..., 2:4] * 2) ** 2 * torch.from_numpy(
                    np.array([self.anchors[idx] for idx in self.anchor_masks[i]])).to(
                    y.device).contiguous().view(1, -1, 1, 1, 2)  # 映射回原图wh尺
                det_13 = (det_1[..., 5:10] * 2) ** 2 * torch.from_numpy(
                    np.array([self.short_anchors[idx] for idx in self.anchor_masks[i]])).to(
                    y.device).contiguous().view(1, -1, 1, 1, 5)  # 映射回原图shortx shorty尺
                det_14 = (det_1[..., 10:13] * 2) ** 2

                det_all = torch.cat((det_11,det_12, det_1[..., [4]], det_13,det_14, det_1[..., 13:], det_2, det_3), dim=4).contiguous()
                detection_bbox_list.append(det_all.view(bs, -1, self.base_num + self.num_classes))
            else:
                det_1 = y.sigmoid()
                det_11 = (det_1[..., 0:2] * 2. - 0.5 + grid_xy) * self.downsample_ratios[i]  # xy 映射回原图位置
                det_12 = (det_1[..., 2:4] * 2) ** 2 * torch.from_numpy(
                    np.array([self.anchors[idx] for idx in self.anchor_masks[i]])).to(
                    y.device).contiguous().view(1, -1, 1, 1, 2)  # 映射回原图wh尺
                
                det_all = torch.cat((det_11, det_12, det_1[...,4:]), dim=4).contiguous()
                detection_bbox_list.append(det_all.view(bs, -1, self.base_num + self.num_classes))

            detection_list.append(det)
        if not self.training:
            detection_bbox_list = torch.cat(detection_bbox_list, 1)
            topk_score, topk_index = torch.topk(detection_bbox_list[..., 4], self.max_num_box)
            detection_bbox_list = detection_bbox_list[:, topk_index.squeeze()]
            return {OUTPUT_DETECTION_YOLO_CONTENT: detection_list, OUTPUT_DETECTION_YOLO_BOX_LIST:detection_bbox_list}
        else:
            detection_bbox_list = torch.cat(detection_bbox_list, 1)
            return {OUTPUT_DETECTION_YOLO_CONTENT: detection_list, OUTPUT_DETECTION_YOLO_BOX_LIST: detection_bbox_list}

    def forward(self, x):
        return self.forward_single(x)

    def forward_train(self, inputs, outputs):
        if inputs.get(COLLECTION_GT_POSITION, None) is not None:
            self.index_bool = inputs[COLLECTION_GT_POSITION] == self.position
            self.index_id = torch.arange(0, inputs[COLLECTION_GT_POSITION].shape[0])[self.index_bool].cpu().numpy().tolist()
        else:
            self.index_id = torch.arange(0, len(inputs[COLLECTION_GT_BBOX_2D])).cpu().numpy().tolist()

        if len(self.index_id) == 0:
            return {}

        # outputs.update(self.forward_single(outputs[OUTPUT_FEATURE]))
        outputs.update(self.forward_single(outputs[OUTPUT_TEACHER_FEATURE_AFFINE]))
        
        if self.with_ref:
            outputs[OUTPUT_DETECTION_YOLO_BOX_LIST_REF] = self.forward_single(outputs[OUTPUT_TEACHER_FEATURE_REF])[OUTPUT_DETECTION_YOLO_BOX_LIST]

        detect_loss = self.loss(inputs, outputs)

        return detect_loss

    def loss(self, inputs, outputs):
        pred_results = outputs[OUTPUT_DETECTION_YOLO_CONTENT]
        raw_gt_bboxes_2d, raw_gt_labels = inputs[COLLECTION_GT_BBOX_2D], inputs[COLLECTION_GT_LABEL]
        
        if self.with_ddd:
            raw_gt_ddd_sizes, raw_gt_ddd_face_clzs, raw_gt_ddd_short_xs, raw_gt_ddd_rotations = inputs[COLLECTION_GT_DDD_SIZES], inputs[COLLECTION_GT_DDD_FACE_CLASSES], inputs[COLLECTION_GT_DDD_SHORT_XS], inputs[COLLECTION_GT_DDD_ROTATIONS]
            raw_gt_ddd_lfs, raw_gt_ddd_rfs, raw_gt_ddd_rrs, raw_gt_ddd_lrs = inputs[COLLECTION_GT_DDD_LFS], inputs[COLLECTION_GT_DDD_RFS], inputs[COLLECTION_GT_DDD_RRS], inputs[COLLECTION_GT_DDD_LRS]

        gt_bboxes_2d, gt_labels, gt_ddd_sizes, gt_ddd_face_clzs, gt_ddd_short_xs, gt_ddd_rotations = [], [], [], [], [], []
        gt_ddd_lfs, gt_ddd_rfs, gt_ddd_rrs, gt_ddd_lrs = [], [], [], []
        
        for i in self.index_id:
            raw_gt_bbox_2d, raw_gt_label = raw_gt_bboxes_2d[i], raw_gt_labels[i]
            dynamic_idx = raw_gt_label < self.num_classes
            gt_bboxes_2d.append(raw_gt_bbox_2d[dynamic_idx])
            gt_labels.append(raw_gt_label[dynamic_idx])

            if self.with_ddd:
                raw_gt_ddd_size, raw_gt_ddd_face_clz, raw_gt_ddd_short_x, raw_gt_ddd_rotation = raw_gt_ddd_sizes[i], raw_gt_ddd_face_clzs[i], raw_gt_ddd_short_xs[i], raw_gt_ddd_rotations[i]
                raw_gt_ddd_lf, raw_gt_ddd_rf, raw_gt_ddd_rr, raw_gt_ddd_lr = raw_gt_ddd_lfs[i], raw_gt_ddd_rfs[i], raw_gt_ddd_rrs[i], raw_gt_ddd_lrs[i]
                gt_ddd_sizes.append(raw_gt_ddd_size[dynamic_idx])
                gt_ddd_face_clzs.append(raw_gt_ddd_face_clz[dynamic_idx])
                gt_ddd_short_xs.append(raw_gt_ddd_short_x[dynamic_idx])
                gt_ddd_rotations.append(raw_gt_ddd_rotation[dynamic_idx])
                gt_ddd_lfs.append(raw_gt_ddd_lf[dynamic_idx])
                gt_ddd_rfs.append(raw_gt_ddd_rf[dynamic_idx])
                gt_ddd_rrs.append(raw_gt_ddd_rr[dynamic_idx])
                gt_ddd_lrs.append(raw_gt_ddd_lr[dynamic_idx])

        indices, tbox, tcls, anchor, tshort, tsize, tsize_anch, tface_cls, trotation_cls, trotation_res, tshort_anch = self.get_target(pred = pred_results, gt_bbox=gt_bboxes_2d, gt_class=gt_labels, gt_ddd_sizes=gt_ddd_sizes, gt_ddd_face_clzs=gt_ddd_face_clzs, gt_ddd_short_xs=gt_ddd_short_xs, gt_ddd_lfs=gt_ddd_lfs, gt_ddd_rfs=gt_ddd_rfs, gt_ddd_rrs=gt_ddd_rrs, gt_ddd_lrs=gt_ddd_lrs, gt_ddd_rotations=gt_ddd_rotations)
        bbox_loss, confidence_loss, class_loss, short_loss, size_loss, face_class_loss, rotation_class_loss, rotation_res_loss = multi_apply(self.loss_single, pred_results, indices, tbox, tcls, anchor, tshort, tsize, tsize_anch, tface_cls, trotation_cls, trotation_res, tshort_anch, self.conf_balances)

        loss = {}
        if self.position:
            loss[f"bbox_loss_{self.position}"] = bbox_loss
            loss[f"confidence_loss_{self.position}"] = confidence_loss
            loss[f"class_loss_{self.position}"] = class_loss
            loss[f"short_loss_{self.position}"] = short_loss
            loss[f"size_loss_{self.position}"] = size_loss
            loss[f"face_class_loss_{self.position}"] = face_class_loss
            loss[f"rotation_class_loss_{self.position}"] = rotation_class_loss
            loss[f"rotation_res_loss_{self.position}"] = rotation_res_loss
        else:
            if self.with_ddd:
                loss[f"short_loss"] = short_loss
                loss[f"size_loss"] = size_loss
                loss[f"face_class_loss"] = face_class_loss
                loss[f"rotation_class_loss"] = rotation_class_loss
                loss[f"rotation_res_loss"] = rotation_res_loss
            loss[f"bbox_loss"] = bbox_loss
            loss[f"confidence_loss"] = confidence_loss
            loss[f"class_loss"] = class_loss
        return loss

    def loss_single(self, pred, indices, tbox, tcls, anchors, tshort, tsize, tsize_anch, tface_cls, trotation_cls, trotation_res, tshort_anch, conf_balances):

        device = pred.device
        ft = torch.cuda.FloatTensor if pred.is_cuda else torch.Tensor
        lcls, lbox, lobj, lshort, lsize, lfcls, lrcls, lrres = ft([0]).to(device), ft([0]).to(device), ft([0]).to(device), ft([0]).to(device), ft([0]).to(device), ft([0]).to(device), ft([0]).to(device), ft([0]).to(device)
        def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
            # return positive, negative label smoothing BCE targets
            return 1.0 - 0.5 * eps, 0.5 * eps

        # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        eps = 0
        if self.label_smooth and self.deta>0:
            eps = self.deta
        cp, cn = smooth_BCE(eps=eps)

        # per output
        nt = 0  # number of targets
        pi = pred
        b, a, gj, gi = indices # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0]).to(device)  # target obj

        nb = b.shape[0]  # number of targets
        if nb:
            nt += nb  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            pxy = ps[:, :2].sigmoid() * 2. - 0.5  # -0.5<pxy<1.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * torch.from_numpy(anchors).to(device)  # 0-4倍缩放 model.hyp['anchor_t']=4
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            bbox_loss, giou= self.bbox_loss(pbox,tbox)
            lbox += bbox_loss

            # Obj
            tobj[b, a, gj, gi] = giou.detach().clamp(0).type(tobj.dtype)  # giou ratio
           
            # Class
            t = torch.full_like(ps[:, self.base_num:], cn).to(device)  # targets
            t[range(nb), tcls.squeeze()] = cp
            lcls += self.class_loss(ps[:, self.base_num:], t)

            if self.with_ddd:
                # ShortX ShortY Res
                pshort = (ps[:, 5:10].sigmoid() * 2) ** 2 * torch.from_numpy(tshort_anch).to(device)
                lshort += self.res_loss(pshort, tshort)
                
                # H W L
                psize = (ps[:, 10:13].sigmoid() * 2) ** 2 * tsize_anch
                lsize += self.res_loss(psize, tsize)

                # Head Face Class
                t = torch.full_like(ps[:, 13:21], cn).to(device)  # targets
                t[range(nb), tface_cls.squeeze()] = cp
                lfcls += self.class_loss(ps[:, 13:21], t)

                # Rotation Class
                index_selected = trotation_cls > 0
                t = torch.full_like(ps[:, 21: 21+self.bins], cn).to(device)  # targets
                t[index_selected] = cp
                lrcls += self.class_loss(ps[:, 21: 21+self.bins], t)

                # Rotation Res
                pred_rotation = ps[:, 21+self.bins:21+3*self.bins].view(-1, self.bins, 2)
                pred_rotation = pred_rotation[index_selected]
                trotation_res = trotation_res[index_selected]
                loss_sin = self.res_loss(pred_rotation[:, 0], torch.sin(trotation_res))
                loss_cos = self.res_loss(pred_rotation[:, 1], torch.cos(trotation_res))
                lrres += loss_sin + loss_cos
            
        lobj += self.confidence_loss(pi[..., 4], tobj) * conf_balances  # obj loss

        return lbox, 5 * lobj, 5 * lcls, lshort, lsize, 5 * lfcls, lrcls, lrres
    
    def res_loss(self, output, target):
        return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

    # def regress_loss(self, regression, targets):
    #     regression_diff = torch.abs(targets - regression)
    #     regression_loss = torch.where(
    #         torch.le(regression_diff, 1.0 / 9.0),
    #         0.5 * 9.0 * torch.pow(regression_diff, 2),
    #         regression_diff - 0.5 / 9.0
    #     )
    #     return regression_loss.mean()

    # 为了简便起见，直接转换数据格式为yolov5 build_target需要的格式，值得注意的是yolov5为了增加训练正样本数量（yolo原始方法一个目标最多由1个grid预测（中心点cx,cy所在位置grid），youlov5采用中心点（cx,cy）所在grid+附件的两个grid一起预测。）
    # 针对预测grid位置判定会根据cx,cy偏移中心点（0.5）位置，多添加两个grid预测位置。在x轴方向cx偏移量>0.5,添加grid+1的位置预测，cx偏移量<0.5,添加grid-1的位置预测。y轴同理
    # 由于添加两个grid预测位置，所以 gt_xy的预测偏移范围为-0.5<pxy<1.5,损失函数定义需要在此范围内。
    # 针对anchor的选择不再是通过IOU找到最合适的anchor方式。二是采用gt框/anchor小于4,(也就是，gt和anchor的缩放比例在4以内都可预测)，预测的缩放尺度为0<pwh<4 ，损失函数定义时候请注意
    # def get_target(self, pred, gt_bbox, gt_class, gt_ddd_sizes, gt_ddd_face_clzs, gt_ddd_short_xs, gt_ddd_short_ys, gt_ddd_rotations):
    def get_target(self, pred, gt_bbox, gt_class, gt_ddd_sizes, gt_ddd_face_clzs, gt_ddd_short_xs, gt_ddd_lfs, gt_ddd_rfs, gt_ddd_rrs, gt_ddd_lrs, gt_ddd_rotations):
        device = pred[0].device
        gain = torch.ones(15+self.bins*2, device=device)  # normalized to gridspace gain
        ft = torch.cuda.FloatTensor if pred[0].is_cuda else torch.Tensor
        targets = ft([]).to(device)
        for i, gtb in enumerate(gt_bbox):
            gtc = gt_class[i].float().view(len(gtb), 1)
            img_idx = torch.ones(len(gtb), 1, device=device) * i
            if self.with_ddd:
                gts = gt_ddd_sizes[i].float().view(len(gtb), 3)
                gtfc = gt_ddd_face_clzs[i].float().view(len(gtb), 1)
                gtsx = gt_ddd_short_xs[i].float().view(len(gtb), 1)
                gtlf = gt_ddd_lfs[i].float().view(len(gtb), 1)
                gtrf = gt_ddd_rfs[i].float().view(len(gtb), 1)
                gtrr = gt_ddd_rrs[i].float().view(len(gtb), 1)
                gtlr = gt_ddd_lrs[i].float().view(len(gtb), 1)
                gtr = gt_ddd_rotations[i].float().view(len(gtb), 1)
                gtrc = torch.zeros((len(gtb), self.bins), device=device).float()
                gtrres = torch.zeros((len(gtb), self.bins), device=device).float()
                for j in range(len(gtb)):
                    for k in range(self.bins):
                        diff = self.centerAngle[k] - gtr[j]
                        diff_2_center = abs(diff)
                        if diff_2_center > np.pi:
                            diff_2_center = 2 * np.pi - diff_2_center
                        if diff_2_center <= self.intervalAngle / 2 + self.overlap:
                            gtrc[j, k] = 1
                            gtrres[j, k] = diff
            else:
                gts = torch.zeros((len(gtb), 3), device=device).float()
                gtfc = torch.zeros((len(gtb), 1), device=device).float()
                gtsx = torch.zeros((len(gtb), 1), device=device).float()
                gtlf = torch.zeros((len(gtb), 1), device=device).float()
                gtrf = torch.zeros((len(gtb), 1), device=device).float()
                gtrr = torch.zeros((len(gtb), 1), device=device).float()
                gtlr = torch.zeros((len(gtb), 1), device=device).float()
                gtr = torch.zeros((len(gtb), 1), device=device).float()
                gtrc = torch.zeros((len(gtb), self.bins), device=device).float()
                gtrres = torch.zeros((len(gtb), self.bins), device=device).float()
            targets = torch.cat((targets, torch.cat((img_idx, gtc, gtb, gtsx, gtlf, gtrf, gtrr, gtlr, gts, gtfc, gtrc, gtrres), dim=-1)))
        na, nt = len(self.anchor_masks), len(targets)
        tcls, tbox, indices, anch, tshort, tsize, tsize_anch, tface_cls, trotation_cls, trotation_res = [], [], [], [], [], [], [], [], [], []
        tshort_anch = []
        targets[..., 2:11] = self.xyxy2xywh(targets[..., 2:11])
        g = 0.5  # offset grid中心偏移
        off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]],
                           device=device).float()  # overlap offsets 按grid区域换算偏移区域， 附近的4个网格 上下左右
        at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)
        for idx, (mask, downsample_ratio) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
            anchors = np.array(self.anchors, dtype=np.float32)[mask] / downsample_ratio  # Scale
            short_anchors = np.array(self.short_anchors, dtype=np.float32)[mask] / downsample_ratio
            if self.with_ddd:
                bbox_size_anchor = torch.from_numpy(np.array(self.bbox_size_anchor, dtype=np.float32)).to(device)  # Scale
            else:
                bbox_size_anchor = torch.zeros((self.num_classes, 3)).to(device)
            gain[2:11] = torch.tensor(pred[idx].shape)[[3, 2, 3, 2, 3, 2, 2, 2, 2]]  # xyxy gain

            # Match targets to anchors
            a, t, offsets = [], targets * gain, 0
            if nt:
                r = t[None, :, 4:6] / torch.from_numpy(anchors[:, None]).to(device)  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
                a, t = at[j], t.repeat(na, 1, 1)[j]  # filter t为过滤后所有匹配锚框缩放尺度小于4的真框 a 位置信息

                # overlaps
                gxy = t[:, 2:4]  # grid xy
                z = torch.zeros_like(gxy)
                # j,k 为小于0.5的偏移 ，l,m为大于0.5的偏移
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]),
                                                                            0)  # t 原始target, t[j] x<.5 偏移的target, t[k] y<.5 偏移的target, t[l] x>.5 偏移的target, t[m] y>.5 偏移的target
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]),
                                    0) * g  # z 原始target,x<0.5 +0.5 ,y<0.5 +0.5,x>.5 -0.5,y>.5 -0.5

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()  # 获取所有的grid 位置 -0.5<offsets<0.5
            gi, gj = gij.T  # grid xy indices
            gshort = t[:, 6:11]
            gsize_anchor = bbox_size_anchor[c]
            gsize = t[:, 11:14]
            gfc = t[:, 14]
            grc = t[:, 15:15+self.bins]
            grres = t[:, 15+self.bins:15+2*self.bins]

            # Append
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box x,y 偏移范围在[-0.5,1.5]
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tshort.append(gshort)
            tsize.append(gsize)
            tsize_anch.append(gsize_anchor)
            tface_cls.append(gfc.long())
            trotation_cls.append(grc.long())
            trotation_res.append(grres)
            tshort_anch.append(short_anchors[a])

            
        return indices, tbox, tcls, anch, tshort, tsize, tsize_anch, tface_cls, trotation_cls, trotation_res, tshort_anch

    def get_nms_result(self, pred):
        det_result = []
        pred = non_max_suppression(pred, self.base_num, conf_thres=self.scores_thr, iou_thres=self.nms_thr, merge=self.merge)
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(self.img_shape, det[:, :4], self.img_shape).round()
                # Write results
                for item in det:
                    object_dic = dict()
                    xyxy = item[:4]
                    object_dic[OUTPUT_DETECTION_DDD_BBOX] = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    object_dic[OUTPUT_DETECTION_DDD_SCORE] = float(item[4])
                    object_dic[OUTPUT_DETECTION_DDD_LABEL] = int(item[5])
                    if self.with_ddd:
                        object_dic[OUTPUT_DETECTION_DDD_SHORTX] = int(item[6])
                        object_dic[OUTPUT_DETECTION_DDD_LF] = int(item[7])
                        object_dic[OUTPUT_DETECTION_DDD_RF] = int(item[8])
                        object_dic[OUTPUT_DETECTION_DDD_RR] = int(item[9])
                        object_dic[OUTPUT_DETECTION_DDD_LR] = int(item[10])
                        box_size = self.bbox_size_anchor[object_dic[OUTPUT_DETECTION_DDD_LABEL]]
                        object_dic[OUTPUT_DETECTION_DDD_H] = float(item[11]*box_size[0])
                        object_dic[OUTPUT_DETECTION_DDD_W] = float(item[12]*box_size[1])
                        object_dic[OUTPUT_DETECTION_DDD_L] = float(item[13]*box_size[2])
                        object_dic[OUTPUT_DETECTION_DDD_HEAD_LABEL] = torch.argmax(item[14:22])
                        object_dic[OUTPUT_DETECTION_DDD_HEAD_SCORE] = item[14 + object_dic[OUTPUT_DETECTION_DDD_HEAD_LABEL]]
                        rotation_label = torch.argmax(item[22:22+self.bins])
                        rotation_score = item[22 + rotation_label]
                        rotation_sin, rotation_cos = item[22+self.bins + rotation_label*2], item[22+self.bins + rotation_label*2 + 1]
                        object_dic[OUTPUT_DETECTION_DDD_ROTATION_CENTER] = float(self.centerAngle[rotation_label])
                        object_dic[OUTPUT_DETECTION_DDD_ROTATION_RES] = float(rotation_sin/rotation_cos)
                    
                    det_result.append(object_dic)

        return det_result

    def xyxy2xywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        y[:, 4] = x[:, 4]
        y[:, 5] = x[:, 5]
        y[:, 6] = x[:, 6]
        y[:, 7] = x[:, 7]
        y[:, 8] = x[:, 8]

        y[:, 0] /= self.img_shape[1]
        y[:, 1] /= self.img_shape[0]
        y[:, 2] /= self.img_shape[1]
        y[:, 3] /= self.img_shape[0]
        y[:, 4] /= self.img_shape[1]
        y[:, 5] /= self.img_shape[0]
        y[:, 6] /= self.img_shape[0]
        y[:, 7] /= self.img_shape[0]
        y[:, 8] /= self.img_shape[0]

        return y


