import torch 
import torch.nn as nn 
import torch.nn.functional as F 


def score_loss(a, b, reduction, alpha=1., eps=1e-8): 
    n = a.shape[0]
    assert a.shape == b.shape == (n,)
    ca = (a.unsqueeze(1) > a).float() # n, n 
    cb = (b.unsqueeze(1) > b).float() # n, n
    sa = ca.sum(-1)
    sb = cb.sum(-1)
    delta = (sa - sb) / (n + eps)
    a2 = a.detach() - delta * alpha
    l = F.l1_loss(a, a2, reduction=reduction)
    return l


class ScoreLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pred, gt, n):
        return score_loss(pred, gt, reduction='none').sum() / n
