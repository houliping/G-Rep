import torch
import torch.nn as nn
from ..registry import LOSSES
from .utils import weighted_loss
from mmdet.ops.gmm import GMM
from mmdet.core.bbox.transforms_kld import gt2gaussian



def kld_single2single(g1, g2):
    p_mu = g1.mu
    p_var = g1.var
    assert p_mu.dim() == 3 and p_mu.size()[1]==1 #(T,1,d)
    assert p_var.dim() == 4 and p_var.size()[1]==1 #(T,1,d,d)
    p_mu = p_mu.squeeze(1)
    p_var = p_var.squeeze(1)
    t_mu, t_var = g2 #(T,d) (T,d,d)
    delta = (p_mu-t_mu).unsqueeze(-1)
    t_inv = torch.inverse(t_var)
    term1 = delta.transpose(-1, -2).matmul(t_inv).matmul(delta).squeeze(-1)
    term2 = torch.diagonal(t_inv.matmul(p_var), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)\
          +torch.log(torch.det(t_var)/torch.det(p_var)).reshape(-1, 1)

    return term1+term2-2 #(T,1)


@weighted_loss
def kld_loss(pred, target, eps=1e-6):
    pred = pred.reshape(-1, 9, 2)
    target = target.reshape(-1, 4, 2)

    # print("# pred: ", pred)
    # print("$ target: ", target)

    assert pred.size()[0] == target.size()[0] and target.numel() > 0
    gmm = GMM(n_components=1, requires_grad=True)
    gmm.fit(pred)
    kld = kld_single2single(gmm, gt2gaussian(target))
    kl_agg = kld.clamp(min=eps)
    # kl_loss = 1 - 1/torch.exp(-(torch.sqrt(kl_agg)))
    kl_loss = 1 - 1/(2+torch.sqrt(kl_agg))
    # kl_loss = torch.sqrt(1 - torch.exp(-(kl_agg)))

    return kl_loss


@LOSSES.register_module
class KLDLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(KLDLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        if weight is not None and not torch.any(weight > 0):
            return (pred * weight.unsqueeze(-1)).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * kld_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox





