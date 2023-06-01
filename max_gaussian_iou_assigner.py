import torch

from ..geometry import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from mmdet.ops.gmm import GMM
from mmdet.core.bbox.transforms_kld import gaussian2bbox


from mmdet.ops.iou import convex_iou

class MaxGaussianIoUAssigner(BaseAssigner):
    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=-1,
                 ):

        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr


    def assign(self, points, gt_rbboxes, overlaps, gt_rbboxes_ignore=None, gt_labels=None, ):
        # print(points.size())
        # print(gt_rbboxes.size())
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
                gt_rbboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = points.device
            bboxes = points.cpu()
            gt_rbboxes = gt_rbboxes.cpu()
            if gt_rbboxes_ignore is not None:
                gt_rbboxes_ignore = gt_rbboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        #      new add
        bboxes = self.points2gaussian2bbox(points)
        points = self.bboxes2points(bboxes)


        if overlaps is None:
            overlaps = self.convex_overlaps(gt_rbboxes, points)
            # print(overlaps.size())

            # overlaps = self.convex_overlaps(gt_rbboxes, points)


        if (self.ignore_iof_thr > 0 and gt_rbboxes_ignore is not None
                and gt_rbboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.convex_overlaps(
                    bboxes, gt_rbboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.convex_overlaps(
                    gt_rbboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_zeros((num_bboxes,),
                                                     dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign fg: for each gt, proposals with highest IoU
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes,))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def points_center_pts(self, pts, y_first=True):

        if y_first:
            pts = pts.reshape(-1, 9, 2)
            pts_dy = pts[:, :, 0::2]
            pts_dx = pts[:, :, 1::2]
            pts_dy_mean = pts_dy.mean(dim=1, keepdim=True).reshape(-1, 1)
            pts_dx_mean = pts_dx.mean(dim=1, keepdim=True).reshape(-1, 1)
            #RPoints = torch.cat([pts_dx_mean, pts_dy_mean], dim=2).reshape(-1, 2)
            dis = torch.pow(torch.pow(pts_dy_mean, 2) + torch.pow(pts_dx_mean, 2), 1/2).reshape(-1, 1)
        else:
            pts = pts.reshape(-1, 9, 2)
            pts_dx = pts[:, :, 0::2]
            pts_dy = pts[:, :, 1::2]
            pts_dy_mean = pts_dy.mean(dim=1, keepdim=True).reshape(-1, 1)
            pts_dx_mean = pts_dx.mean(dim=1, keepdim=True).reshape(-1, 1)
            #RPoints = torch.cat([pts_dx_mean, pts_dy_mean], dim=2).reshape(-1, 2)
            dis = torch.pow(torch.pow(pts_dy_mean, 2) + torch.pow(pts_dx_mean, 2), 1/2).reshape(-1, 1)
        return dis

    def points2gaussian2bbox(self, points):
        points = points.reshape(-1, 9, 2)
        gmm = GMM(n_components=1, requires_grad=True)
        gmm.fit(points)
        bbox = gaussian2bbox(gmm)
        return bbox



    def convex_overlaps(self, gt_rbboxes, points):
        overlaps = convex_iou(points, gt_rbboxes)
        overlaps = overlaps.transpose(1, 0)
        return overlaps

    def bboxes2points(self, bboxes):
        bboxes = bboxes.reshape(-1, 4, 2)
        bboxes_dx = bboxes[:, :, 0::2].reshape(-1, 4)
        bboxes_dy = bboxes[:, :, 1::2].reshape(-1, 4)

        bboxes_dx_mean = bboxes_dx.mean(dim=1, keepdim=True).reshape(-1, 1)
        bboxes_dy_mean = bboxes_dy.mean(dim=1, keepdim=True).reshape(-1, 1)

        bboxes_x01 = ((bboxes_dx[:, 0] + bboxes_dx[:, 1]) / 2).reshape(-1, 1)
        bboxes_x12 = ((bboxes_dx[:, 1] + bboxes_dx[:, 2]) / 2).reshape(-1, 1)
        bboxes_x23 = ((bboxes_dx[:, 2] + bboxes_dx[:, 3]) / 2).reshape(-1, 1)
        bboxes_x30 = ((bboxes_dx[:, 3] + bboxes_dx[:, 0]) / 2).reshape(-1, 1)

        bboxes_y01 = ((bboxes_dy[:, 0] + bboxes_dy[:, 1]) / 2).reshape(-1, 1)
        bboxes_y12 = ((bboxes_dy[:, 1] + bboxes_dy[:, 2]) / 2).reshape(-1, 1)
        bboxes_y23 = ((bboxes_dy[:, 2] + bboxes_dy[:, 3]) / 2).reshape(-1, 1)
        bboxes_y30 = ((bboxes_dy[:, 3] + bboxes_dy[:, 0]) / 2).reshape(-1, 1)

        points = torch.cat([bboxes_dx[:, 0].reshape(-1, 1), bboxes_dy[:, 0].reshape(-1, 1),
                            bboxes_dx[:, 1].reshape(-1, 1), bboxes_dy[:, 1].reshape(-1, 1),
                            bboxes_dx[:, 2].reshape(-1, 1), bboxes_dy[:, 2].reshape(-1, 1),
                            bboxes_dx[:, 3].reshape(-1, 1), bboxes_dy[:, 3].reshape(-1, 1),
                            bboxes_x01, bboxes_y01,
                            bboxes_x12, bboxes_y12,
                            bboxes_x23, bboxes_y23,
                            bboxes_x30, bboxes_y30,
                            bboxes_dx_mean, bboxes_dy_mean], dim=1).reshape(-1, 18)

        return points




