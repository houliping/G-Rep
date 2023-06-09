from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.cnn import constant_init
from mmdet.core import (PointGenerator, multi_apply, multiclass_rnms,
                        images_to_levels, unmap)
from mmdet.core import (ConvexPseudoSampler, assign_and_sample, build_assigner)
from mmdet.ops import ConvModule, DeformConv
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob
from mmdet.ops.minareabbox import find_minarea_rbbox
from mmdet.ops.iou import convex_iou
from mmdet.ops.gmm import GMM

INF = 100000000
eps = 1e-12


def levels_to_images(mlvl_tensor, flatten=False):
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    if flatten:
        channels = mlvl_tensor[0].size(-1)
    else:
        channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        if not flatten:
            t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]

def gaussian2bbox(gmm):
    var=gmm.var
    mu=gmm.mu
    assert mu.size()[1:]==(1,2) # (T,1,2)
    assert var.size()[1:]==(1,2,2) # (T,1,2,2)
    T=mu.size()[0]
    var=var.squeeze(1)
    U,s,Vt=torch.svd(var)
    # print(s.size())
    size_half=2*s.sqrt().unsqueeze(1).repeat(1,4,1) # (T,4,2)
    mu=mu.repeat(1,4,1) # (T,4,2)
    dx_dy=size_half*torch.tensor([[-1,1],
                                  [1,1],
                                  [1,-1],
                                  [-1,-1]],
                                 dtype=torch.float32,device=size_half.device) # (T,4,2)
    # assert (U.transpose(-1,-2)==Vt).all()
    bboxes=(mu+dx_dy.matmul(Vt)).reshape(T,8)
    return bboxes


@HEADS.register_module
class ATSSRepPointsHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 point_feat_channels=256,
                 stacked_convs=3,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init=dict(
                     type='SmoothL1Loss', loss_weight=0.375),
                 loss_bbox_refine=dict(
                     type='SmoothL1Loss', loss_weight=0.75),
                 center_init=True,
                 transform_method='rotrect',
                 show_points=False,
                 topk=6,
                 anti_factor=0.75):

        super(ATSSRepPointsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.num_points = num_points
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.center_init = center_init
        self.transform_method = transform_method
        self.show_points = show_points
        self.topk = topk
        self.anti_factor = anti_factor

        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes
        self.point_generators = [PointGenerator() for _ in self.point_strides]
        # we use deformable conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        pts_out_dim = 2 * self.num_points
        self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                             self.point_feat_channels,
                                             self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv(self.feat_channels,
                                                    self.point_feat_channels,
                                                    self.dcn_kernel, 1,
                                                    self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)

    def points_center_pts(self, pts, y_first=True):

        if y_first:
            pts = pts.reshape(-1, self.num_points, 2)
            pts_dy = pts[:, :, 0::2]
            pts_dx = pts[:, :, 1::2]
            pts_dy_mean = pts_dy.mean(dim=1, keepdim=True).reshape(-1, 1)
            pts_dx_mean = pts_dx.mean(dim=1, keepdim=True).reshape(-1, 1)
            # RPoints = torch.cat([pts_dx_mean, pts_dy_mean], dim=2).reshape(-1, 2)
            dis = torch.pow(torch.pow(pts_dy_mean, 2) + torch.pow(pts_dx_mean, 2), 1 / 2).reshape(-1, 1)
        else:
            pts = pts.reshape(-1, self.num_points, 2)
            pts_dx = pts[:, :, 0::2]
            pts_dy = pts[:, :, 1::2]
            pts_dy_mean = pts_dy.mean(dim=1, keepdim=True).reshape(-1, 1)
            pts_dx_mean = pts_dx.mean(dim=1, keepdim=True).reshape(-1, 1)
            # RPoints = torch.cat([pts_dx_mean, pts_dy_mean], dim=2).reshape(-1, 2)
            dis = torch.pow(torch.pow(pts_dy_mean, 2) + torch.pow(pts_dx_mean, 2), 1 / 2).reshape(-1, 1)
        return dis

    def convex_overlaps(self, gt_rbboxes, points):
        overlaps = convex_iou(points, gt_rbboxes)

        # iou_weight = torch.pow(torch.exp(-1 * self.points_center_pts(points)), 1/4)
        # overlaps = overlaps * iou_weight

        overlaps = overlaps.transpose(1, 0)  # [gt, ex]
        return overlaps



    def points2rotrect(self, pts, y_first=True):
        if y_first:
            pts = pts.reshape(-1, self.num_points, 2)
            pts_dy = pts[:, :, 0::2]
            pts_dx = pts[:, :, 1::2]
            pts = torch.cat([pts_dx, pts_dy], dim=2).reshape(-1, 2 * self.num_points)

        if self.transform_method == 'rotrect':
            rotrect_pred = find_minarea_rbbox(pts)
            return rotrect_pred
        else:
            raise NotImplementedError

    def forward_single(self, x):
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        # If we use center_init, the initial reppoints is from center points.
        # If we use bounding bbox representation, the initial reppoints is
        #   from regular grid placed on a pre-defined bbox.
        points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset)
        cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))
        pts_out_refine = self.reppoints_pts_refine_out(self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))

        pts_out_refine = pts_out_refine + pts_out_init.detach()
        return cls_out, pts_out_init, pts_out_refine

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_points(self, featmap_sizes, img_metas):
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i])
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]
        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return points_list, valid_flag_list

    def offset_to_pts(self, center_list, pred_list):
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)

                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine, labels,
                    label_weights, rbbox_gt_init, convex_weights_init,
                    rbbox_gt_refine, convex_weights_refine, stride, num_total_samples_refine):
        normalize_term = self.point_base_scale * stride

        rbbox_gt_init = rbbox_gt_init.reshape(-1, 8)
        convex_weights_init = convex_weights_init.reshape(-1)
        # init points loss
        pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)  # [B, NUM(H * W), 2*num_pint]
        pos_ind_init = (convex_weights_init > 0).nonzero().reshape(-1)
        pts_pred_init_norm = pts_pred_init[pos_ind_init]
        rbbox_gt_init_norm = rbbox_gt_init[pos_ind_init]
        convex_weights_pos_init = convex_weights_init[pos_ind_init]
        loss_pts_init = self.loss_bbox_init(
            pts_pred_init_norm / normalize_term,
            rbbox_gt_init_norm / normalize_term,
            convex_weights_pos_init
        )
        # refine points loss
        rbbox_gt_refine = rbbox_gt_refine.reshape(-1, 8)
        pts_pred_refine = pts_pred_refine.reshape(-1, 2 * self.num_points)
        convex_weights_refine = convex_weights_refine.reshape(-1)
        pos_ind_refine = (convex_weights_refine > 0).nonzero().reshape(-1)
        pts_pred_refine_norm = pts_pred_refine[pos_ind_refine]
        rbbox_gt_refine_norm = rbbox_gt_refine[pos_ind_refine]
        convex_weights_pos_refine = convex_weights_refine[pos_ind_refine]
        loss_pts_refine = self.loss_bbox_refine(
            pts_pred_refine_norm / normalize_term,
            rbbox_gt_refine_norm / normalize_term,
            convex_weights_pos_refine
        )
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score,
            labels,
            label_weights,
            avg_factor=num_total_samples_refine)
        return loss_cls, loss_pts_init, loss_pts_refine



    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             gt_rbboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_rbboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)
        if cfg.init.assigner['type'] == 'ConvexAssigner':
            candidate_list = center_list
        else:
            raise NotImplementedError
        cls_reg_targets_init = self.point_target(
            candidate_list,
            valid_flag_list,
            gt_rbboxes,
            img_metas,
            cfg.init,
            gt_rbboxes_ignore_list=gt_rbboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        (*_, rbbox_gt_list_init, candidate_list_init, convex_weights_list_init,
         num_total_pos_init, num_total_neg_init, gt_inds_init) = cls_reg_targets_init
        num_total_samples_init = (num_total_pos_init +
                                  num_total_neg_init if self.sampling else num_total_pos_init)
        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        pts_coordinate_preds_refine = self.offset_to_pts(center_list, pts_preds_refine)
        points_list = []
        for i_img, center in enumerate(center_list):
            points = []
            for i_lvl in range(len(pts_preds_refine)):
                points_preds_init_ = pts_preds_init[i_lvl].detach()
                points_preds_init_ = points_preds_init_.view(points_preds_init_.shape[0], -1,
                                                             *points_preds_init_.shape[2:])
                points_shift = points_preds_init_.permute(0, 2, 3, 1) * self.point_strides[i_lvl]
                points_center = center[i_lvl][:, :2].repeat(1, self.num_points)
                points.append(points_center + points_shift[i_img].reshape(-1, 2 * self.num_points))
            points_list.append(points)

        cls_reg_targets_refine = self.point_target_refine(
            points_list,
            valid_flag_list,
            gt_rbboxes,
            img_metas,
            cfg.refine,
            gt_rbboxes_ignore_list=gt_rbboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            featmap_sizes=featmap_sizes)
        (labels_list, label_weights_list, rbbox_gt_list_refine,
         candidate_list_refine, convex_weights_list_refine, num_total_pos_refine,
         num_total_neg_refine, gt_inds_refine) = cls_reg_targets_refine
        num_total_samples_refine = (
            num_total_pos_refine +
            num_total_neg_refine if self.sampling else num_total_pos_refine)

        losses_cls, losses_pts_init, losses_pts_refine = multi_apply(
            self.loss_single,
            cls_scores,
            pts_coordinate_preds_init,
            pts_coordinate_preds_refine,
            labels_list,
            label_weights_list,
            rbbox_gt_list_init,
            convex_weights_list_init,
            rbbox_gt_list_refine,
            convex_weights_list_refine,
            self.point_strides,
            num_total_samples_refine=num_total_samples_refine
        )
        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_pts_init': losses_pts_init,
            'loss_pts_refine': losses_pts_refine
        }
        return loss_dict_all



    def get_pos_loss(self, cls_score, pts_pred, label, rbbox_gt,
                     label_weight, convex_weight, pos_inds):
        pos_scores = cls_score[pos_inds]
        pos_pts_pred = pts_pred[pos_inds]
        pos_rbbox_gt = rbbox_gt[pos_inds]
        pos_label = label[pos_inds]
        pos_label_weight = label_weight[pos_inds]
        pos_convex_weight = convex_weight[pos_inds]
        loss_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        loss_bbox = self.loss_bbox_refine(
            pos_pts_pred,
            pos_rbbox_gt,
            pos_convex_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        loss_cls = loss_cls.sum(-1)
        pos_loss = loss_bbox + loss_cls
        return pos_loss,


    def point_target(self,
                     proposals_list,
                     valid_flag_list,
                     gt_rbboxes_list,
                     img_metas,
                     cfg,
                     gt_rbboxes_ignore_list=None,
                     gt_labels_list=None,
                     label_channels=1,
                     sampling=True,
                     unmap_outputs=True,
                     featmap_sizes=None):

        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs
        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]
        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
        # compute targets for each image
        if gt_rbboxes_ignore_list is None:
            gt_rbboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_overlaps_rotate_list = [None] * 4
        (all_labels, all_label_weights, all_rbbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list, all_gt_inds_list) = multi_apply(
            self.point_target_single,
            proposals_list,
            valid_flag_list,
            gt_rbboxes_list,
            gt_rbboxes_ignore_list,
            gt_labels_list,
            all_overlaps_rotate_list,
            cfg=cfg,
            label_channels=label_channels,
            sampling=sampling,
            unmap_outputs=unmap_outputs)
        # no valid points
        if any([labels is None for labels in all_labels]):
            return None
        # sampled points of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_proposals)
        rbbox_gt_list = images_to_levels(all_rbbox_gt, num_level_proposals)
        proposals_list = images_to_levels(all_proposals, num_level_proposals)
        proposal_weights_list = images_to_levels(all_proposal_weights,
                                                 num_level_proposals)
        gt_inds_list = images_to_levels(all_gt_inds_list, num_level_proposals)
        return (labels_list, label_weights_list, rbbox_gt_list, proposals_list,
                proposal_weights_list, num_total_pos, num_total_neg, gt_inds_list)

    def point_target_single(self,
                            flat_proposals,
                            valid_flags,
                            gt_rbboxes,
                            gt_rbboxes_ignore,
                            gt_labels,
                            overlaps,
                            cfg,
                            label_channels=1,
                            sampling=True,
                            unmap_outputs=True):
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]
        if sampling:
            assign_result, sampling_result = assign_and_sample(
                proposals, gt_rbboxes, gt_rbboxes_ignore, None, cfg)
        else:
            bbox_assigner = build_assigner(cfg.assigner)
            assign_result = bbox_assigner.assign(proposals, gt_rbboxes, overlaps,
                                                 gt_rbboxes_ignore, gt_labels)
            bbox_sampler = ConvexPseudoSampler()
            sampling_result = bbox_sampler.sample(assign_result, proposals,
                                                  gt_rbboxes)
        gt_inds = assign_result.gt_inds
        num_valid_proposals = proposals.shape[0]
        rbbox_gt = proposals.new_zeros([num_valid_proposals, 8])
        pos_proposals = torch.zeros_like(proposals)
        proposals_weights = proposals.new_zeros(num_valid_proposals)
        labels = proposals.new_zeros(num_valid_proposals, dtype=torch.long)
        label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_gt_rbboxes = sampling_result.pos_gt_rbboxes
            rbbox_gt[pos_inds, :] = pos_gt_rbboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals, inside_flags)
            rbbox_gt = unmap(rbbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals, inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)
            gt_inds = unmap(gt_inds, num_total_proposals, inside_flags)

        return (labels, label_weights, rbbox_gt, pos_proposals, proposals_weights,
                pos_inds, neg_inds, gt_inds)

    def point_target_refine(self,
                            proposals_list,
                            valid_flag_list,
                            gt_rbboxes_list,
                            img_metas,
                            cfg,
                            gt_rbboxes_ignore_list=None,
                            gt_labels_list=None,
                            label_channels=1,
                            sampling=True,
                            unmap_outputs=True,
                            featmap_sizes=None):

        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs
        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]
        num_level_proposals_list = [num_level_proposals] * num_imgs

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
        # compute targets for each image
        if gt_rbboxes_ignore_list is None:
            gt_rbboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_overlaps_rotate_list = [None] * 4
        (all_labels, all_label_weights, all_rbbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list, all_gt_inds_list) = multi_apply(
            self.point_target_refine_single,
            proposals_list,
            num_level_proposals_list,
            valid_flag_list,
            gt_rbboxes_list,
            gt_rbboxes_ignore_list,
            gt_labels_list,
            all_overlaps_rotate_list,
            cfg=cfg,
            label_channels=label_channels,
            sampling=sampling,
            unmap_outputs=unmap_outputs)
        # no valid points
        if any([labels is None for labels in all_labels]):
            return None
        # sampled points of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_proposals)
        rbbox_gt_list = images_to_levels(all_rbbox_gt, num_level_proposals)
        proposals_list = images_to_levels(all_proposals, num_level_proposals)
        proposal_weights_list = images_to_levels(all_proposal_weights,
                                                 num_level_proposals)
        gt_inds_list = images_to_levels(all_gt_inds_list, num_level_proposals)
        return (labels_list, label_weights_list, rbbox_gt_list, proposals_list,
                proposal_weights_list, num_total_pos, num_total_neg, gt_inds_list)

    def point_target_refine_single(self,
                                   flat_proposals,
                                   num_level_proposals,
                                   valid_flags,
                                   gt_rbboxes,
                                   gt_rbboxes_ignore,
                                   gt_labels,
                                   overlaps,
                                   cfg,
                                   label_channels=1,
                                   sampling=True,
                                   unmap_outputs=True):
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]  # torch.Size([34100, 18])

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_proposals, inside_flags)

        # print('proposals', proposals.shape)
        if sampling:
            assign_result, sampling_result = assign_and_sample(
                proposals, gt_rbboxes, gt_rbboxes_ignore, None, cfg)
        else:
            bbox_assigner = build_assigner(cfg.assigner)
            if cfg.assigner.type != "ATSSAssigner" and cfg.assigner.type != "ATSSConvexAssigner" \
                    and cfg.assigner.type != "LAConvexAssigner" and cfg.assigner.type != "ATSSKldAssigner":
                assign_result = bbox_assigner.assign(proposals, gt_rbboxes, overlaps,
                                                     gt_rbboxes_ignore, gt_labels)
            else:
                assign_result = bbox_assigner.assign(proposals, num_level_anchors_inside, gt_rbboxes,
                                                     gt_rbboxes_ignore, gt_labels)

            bbox_sampler = ConvexPseudoSampler()
            sampling_result = bbox_sampler.sample(assign_result, proposals,
                                                  gt_rbboxes)
        gt_inds = assign_result.gt_inds
        num_valid_proposals = proposals.shape[0]
        rbbox_gt = proposals.new_zeros([num_valid_proposals, 8])
        pos_proposals = torch.zeros_like(proposals)  # torch.Size([34100, 18])9point
        proposals_weights = proposals.new_zeros(num_valid_proposals)
        labels = proposals.new_zeros(num_valid_proposals, dtype=torch.long)
        label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_gt_rbboxes = sampling_result.pos_gt_rbboxes
            rbbox_gt[pos_inds, :] = pos_gt_rbboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        # print('proposals_weights', proposals_weights.shape, proposals_weights)

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals, inside_flags)
            rbbox_gt = unmap(rbbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals, inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)
            gt_inds = unmap(gt_inds, num_total_proposals, inside_flags)

        return (labels, label_weights, rbbox_gt, pos_proposals, proposals_weights,
                pos_inds, neg_inds, gt_inds)

    def get_bboxes(self,
                   cls_scores,
                   pts_preds_init,
                   pts_preds_refine,
                   img_metas,
                   cfg,
                   rescale=False,
                   nms=True):
        assert len(cls_scores) == len(pts_preds_refine)

        num_levels = len(cls_scores)
        mlvl_points = [
            self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            points_pred_list = [
                pts_preds_refine[i][img_id].detach()
                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, points_pred_list,
                                               mlvl_points, img_shape,
                                               scale_factor, cfg, rescale, nms)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          points_preds,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False,
                          nms=True):
        assert len(cls_scores) == len(points_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        if self.show_points:
            mlvl_reppoints = []

        for i_lvl, (cls_score, points_pred, points) in enumerate(
                zip(cls_scores, points_preds, mlvl_points)):
            assert cls_score.size()[-2:] == points_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            points_pred = points_pred.permute(1, 2, 0).reshape(-1, 2 * self.num_points)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                points_pred = points_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bbox_pred = self.points2rotrect(points_pred)

            bbox_pos_center = points[:, :2].repeat(1, 4)
            bboxes = bbox_pred * self.point_strides[i_lvl] + bbox_pos_center

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            if self.show_points:
                points_pred = points_pred.reshape(-1, self.num_points, 2)
                points_pred_dy = points_pred[:, :, 0::2]
                points_pred_dx = points_pred[:, :, 1::2]
                pts = torch.cat([points_pred_dx, points_pred_dy], dim=2).reshape(-1, 2 * self.num_points)

                pts_pos_center = points[:, :2].repeat(1, self.num_points)
                pts = pts * self.point_strides[i_lvl] + pts_pos_center

                mlvl_reppoints.append(pts)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if self.show_points:
            mlvl_reppoints = torch.cat(mlvl_reppoints)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_reppoints /= mlvl_reppoints.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        if nms:
            det_bboxes, det_labels = multiclass_rnms(mlvl_bboxes, mlvl_scores,
                                                     cfg.score_thr, cfg.nms,
                                                     cfg.max_per_img,
                                                     multi_reppoints=mlvl_reppoints if self.show_points else None)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
