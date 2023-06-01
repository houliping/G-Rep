import torch

from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from mmdet.ops.gmm import GMM
from mmdet.core.bbox.transforms_kld import gt2gaussian

class MaxBcdAssigner(BaseAssigner):
    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=-1,
                 mode="threshold",
                 topk=9):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr

    def assign(self, points, gt_rbboxes, overlaps, gt_rbboxes_ignore=None, gt_labels=None, ):
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
                gt_rbboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = points.device
            points = points.cpu()
            gt_rbboxes = gt_rbboxes.cpu()
            if gt_rbboxes_ignore is not None:
                gt_rbboxes_ignore = gt_rbboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        points = points.reshape(-1,9,2) #(N,9,2)
        gt_rbboxes = gt_rbboxes.reshape(-1,4,2) #(K,4,2)
        gmm = GMM(n_components=1)
        gmm.fit(points)

        bcd = self.bcd_mixture2single(gmm, gt2gaussian(gt_rbboxes)) #(K,N)
        overlaps = 1 / (1 + bcd * bcd)

        if (self.ignore_iof_thr > 0 and gt_rbboxes_ignore is not None
                and gt_rbboxes_ignore.numel() > 0 and points.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = 1 / (1 + torch.pow(self.bcd_mixture2single(gmm, gt2gaussian(gt_rbboxes_ignore), 2)))
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = 1 / (1 + torch.pow(self.bcd_mixture2single(gmm, gt2gaussian(gt_rbboxes_ignore), 2)))
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
        num_gts, num_pointsets = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_pointsets,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_pointsets == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_pointsets,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_zeros((num_pointsets,),
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
            assigned_labels = assigned_gt_inds.new_zeros((num_pointsets,))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def bcd_mixture2single(self, g1, g2):
        p_mu = g1.mu
        p_var = g1.var
        assert p_mu.dim() == 3 and p_mu.size()[1] == 1  # (N,1,d)
        assert p_var.dim() == 4 and p_var.size()[1] == 1  # (N,1,d,d)
        N, _, d = p_mu.shape
        p_mu = p_mu.reshape(1, N, d)
        p_var = p_var.reshape(1, N, d, d)

        t_mu, t_var = g2
        K = t_mu.shape[0]
        t_mu = t_mu.unsqueeze(1)  # (K,1,d)
        t_var = t_var.unsqueeze(1)  # (K,1,d,d)

        delta = (p_mu - t_mu).unsqueeze(-1)  # (K,N,d,1)

        var_sum = 0.5 * (p_var + t_var)
        var_sum_inv = torch.inverse(var_sum)

        term1 = torch.log(torch.det(var_sum) / (torch.sqrt(torch.det(t_var.matmul(p_var)))))
        term2 = delta.transpose(-1, -2).matmul(var_sum_inv).matmul(delta).reshape(K, N)
        return 0.5 * term1 + 0.25 * term2  # (T,1)


