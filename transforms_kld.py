
import math

import numpy as np
import torch
from mmdet.ops.gmm import GMM
from mmdet.ops.torch_batch_svd import svd

L=3
def gaussian2bbox(gmm):
    var = gmm.var
    mu = gmm.mu
    assert mu.size()[1:] == (1, 2)                #(T,1,2)
    assert var.size()[1:] == (1, 2, 2)           #(T,1,2,2)
    T = mu.size()[0]
    var = var.squeeze(1)
    U, s, Vt = svd(var)
    # U, s, Vt = torch.svd(var)
    # print(s.size())
    size_half = L * s.sqrt().unsqueeze(1).repeat(1, 4, 1)           #(T,4,2)
    mu = mu.repeat(1, 4, 1)                     #(T,4,2)
    dx_dy = size_half*torch.tensor([[-1, 1],
                                  [1, 1],
                                  [1, -1],
                                  [-1, -1]],
                                 dtype=torch.float32, device=size_half.device) # (T,4,2)
    # bboxes = (mu+dx_dy.matmul(Vt)).reshape(T, 8)
    bboxes = (mu + dx_dy.matmul(Vt.transpose(1, 2))).reshape(T, 8)

    return bboxes



def gt2gaussian(target):
    center = torch.mean(target, dim=1)
    edge_1 = target[:, 1, :] - target[:, 0, :]
    edge_2 = target[:, 2, :] - target[:, 1, :]
    w = (edge_1 * edge_1).sum(dim=-1, keepdim=True)
    w_ = w.sqrt()
    h = (edge_2 * edge_2).sum(dim=-1, keepdim=True)
    diag = torch.cat([w, h], dim=-1).diag_embed() / 4 * L * L
    # cos_=edge_1[:,0].reshape(-1,1)/w_
    # sin_=edge_1[:,1].reshape(-1,1)/w_
    # R=torch.cat([cos_,cos_],dim=-1).diag_embed()+torch.cat([-sin_,sin_],dim=-1).diag_embed()[...,[1,0]]
    cos_sin = edge_1 / w_
    neg = torch.tensor([[1, -1]], dtype=torch.float32).to(cos_sin.device)
    R = torch.stack([cos_sin * neg, cos_sin[..., [1, 0]]], dim=-2)

    return (center, R.matmul(diag).matmul(R.transpose(-1, -2)))  # (K,d) (K,d,d)