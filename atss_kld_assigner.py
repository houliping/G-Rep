import torch

from ..geometry import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from mmdet.ops.iou import convex_iou
from mmdet.ops.point_justify import pointsJf
from mmdet.ops.gmm import GMM
from mmdet.ops.minareabbox import find_minarea_rbbox
from mmdet.ops.iou import convex_giou
# from mmdet.ops.chamfer_2d import Chamfer2D
from mmdet.core.bbox.transforms_kld import gt2gaussian

eps=1e-6

class ATSSKldAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level



    """

    def __init__(self, topk, use_sa=False):
        self.topk = topk
        self.use_sa = use_sa

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5.  compute the mean aspect ratio of all gts, and set exp((-mean aspect ratio / 4) * (mean + std) as the iou threshold
        6. select these candidates whose iou are greater than or equal to
           the threshold as postive
        7. limit the positive sample's center in gt


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        overlaps = self.kld_overlaps(gt_bboxes, bboxes)  # (gt_bboxes, points)
        overlaps = overlaps.transpose(1, 0)
        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_zeros((num_bboxes, ),
                                                     dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        # the center of poly
        gt_bboxes_hbb = self.get_horizontal_bboxes(gt_bboxes)  #convert to hbb

        gt_cx = (gt_bboxes_hbb[:, 0] + gt_bboxes_hbb[:, 2]) / 2.0
        gt_cy = (gt_bboxes_hbb[:, 1] + gt_bboxes_hbb[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        #calculat center of points
        #y_first True or False?

        '''
        (1)points to rbox 
        (2)calculate the center of rbox
        # print('predicted_points', bboxes.shape)
        rboxes = find_minarea_rbbox(bboxes)
        # print('rboxes', rboxes.shape)
        rbboxes_cx = (rboxes[:, 0] + rboxes[:, 4]) / 2.0
        rbboxes_cy = (rboxes[:, 1] + rboxes[:, 5]) / 2.0
        bboxes_points = torch.stack((rbboxes_cx, rbboxes_cy), dim=1)
        # rbboxes_cx_1 = (rboxes[:, 2] + rboxes[:, 6]) / 2.0
        # rbboxes_cy_1 = (rboxes[:, 3] + rboxes[:, 7]) / 2.0
        # print(torch.abs(rbboxes_cx - pts_x_mean).sum())
        # print(torch.abs(rbboxes_cy - pts_y_mean).sum())
        '''

        bboxes = bboxes.reshape(-1, 9, 2)
        # y_first False
        pts_x = bboxes[:, :, 0::2]  #
        pts_y = bboxes[:, :, 1::2]  #

        pts_x_mean = pts_x.mean(dim=1).squeeze()
        pts_y_mean = pts_y.mean(dim=1).squeeze()
        bboxes_points = torch.stack((pts_x_mean, pts_y_mean), dim=1)


        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(
                self.topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        # add

        # print(gt_bboxes[candidate_idxs])
        gt_bboxes_ratios = self.AspectRatio(gt_bboxes)
        gt_bboxes_ratios_per_gt = gt_bboxes_ratios.mean(0)
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        if self.use_sa:
            iou_thr_weight = torch.exp((-1 / 4) * gt_bboxes_ratios_per_gt)
            overlaps_thr_per_gt = overlaps_thr_per_gt * iou_thr_weight



        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]


        inside_flag = torch.full([num_bboxes, num_gt], 0.).to(gt_bboxes.device).float()
        pointsJf(bboxes_points, \
                 gt_bboxes,\
                inside_flag)
        is_in_gts = inside_flag[candidate_idxs, torch.arange(num_gt)].to(is_pos.dtype)

        '''
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes

        ep_bboxes_cx = pts_x_mean.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = pts_y_mean.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)
        # print('candidate_idxs', candidate_idxs.shape, candidate_idxs)

        # calculate the left, top, right, bottom distance between positive bbox center and gt side
        # print('ep_bboxes_cx[candidate_idxs].view(-1, num_gt)', ep_bboxes_cx[candidate_idxs].view(-1, num_gt).shape, ep_bboxes_cx[candidate_idxs].view(-1, num_gt))
        # print('ep_bboxes_cy[candidate_idxs].view(-1, num_gt)', ep_bboxes_cy[candidate_idxs].view(-1, num_gt).shape, ep_bboxes_cx[candidate_idxs].view(-1, num_gt))

        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes_hbb[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes_hbb[:, 1]
        r_ = gt_bboxes_hbb[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes_hbb[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        # print('l_', l_.shape) #[45, 2] 2代表有几个gt box
        # print('gt_bboxes', gt_bboxes.shape)# [2, 4]
        # print()
        # print('torch.stack([l_, t_, r_, b_], dim=1)', torch.stack([l_, t_, r_, b_], dim=1).shape)#在第1维升了一维，进行合并
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01 #最小值大于0.01,也就是全部大于0.01,也就是中心点在框里面。
        # print('is_in_gts', is_in_gts.shape) # [45, 2]
        '''

        is_pos = is_pos & is_in_gts
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)



    def kld_overlaps(self, gt_rbboxes, points):
        points = points.reshape(-1, 9, 2)  # (N,9,2)
        gt_rbboxes = gt_rbboxes.reshape(-1, 4, 2)  # (K,4,2)
        gmm = GMM(n_components=1)
        gmm.fit(points)
        kld = self.kld_mixture2single(gmm, gt2gaussian(gt_rbboxes))  # (K,N)
        kl_agg = kld.clamp(min=eps)
        overlaps = 1 / (2 + kl_agg)

        #  need to debug
        # overlaps = 1 / (2 + torch.sqrt(kld))
        # overlaps = 1 -  torch.exp(-(torch.sqrt(kl_agg)))
        return overlaps




    def kld_mixture2single(self, g1, g2):
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
        t_inv = torch.inverse(t_var)  # (K,1,d,d)
        term1 = delta.transpose(-1, -2).matmul(t_inv).matmul(delta).reshape(K, N)  # (K,N)
        term2 = torch.diagonal(t_inv.matmul(p_var), dim1=-2, dim2=-1).sum(dim=-1) \
                + torch.log(torch.det(t_var) / torch.det(p_var))  # (K,N)

        return term1 + term2 - 2  # (K,N)


    def get_horizontal_bboxes(self, gt_rbboxes):
        gt_xs, gt_ys = gt_rbboxes[:, 0::2], gt_rbboxes[:, 1::2]
        gt_xmin, _ = gt_xs.min(1)
        gt_ymin, _ = gt_ys.min(1)
        gt_xmax, _ = gt_xs.max(1)
        gt_ymax, _ = gt_ys.max(1)
        gt_rect_bboxes = torch.cat([gt_xmin[:, None], gt_ymin[:, None],
                                    gt_xmax[:, None], gt_ymax[:, None]], dim=1)
        return gt_rect_bboxes

    def AspectRatio(self, gt_rbboxes):
        # gt_rbboxes = torch.squeeze(gt_rbboxes)
        # print('AspectRatio.gt_rbboxes')
        # print(gt_rbboxes.size())

        pt1, pt2, pt3, pt4 = gt_rbboxes[..., :8].chunk(4, 1)

        edge1 = torch.sqrt(
            torch.pow(pt1[..., 0] - pt2[..., 0], 2) + torch.pow(pt1[..., 1] - pt2[..., 1], 2))
        edge2 = torch.sqrt(
            torch.pow(pt2[..., 0] - pt3[..., 0], 2) + torch.pow(pt2[..., 1] - pt3[..., 1], 2))

        edges = torch.stack([edge1, edge2], dim=1)
        width, _ = torch.max(edges, 1)
        height, _ = torch.min(edges, 1)
        ratios = (width / height)
        return ratios