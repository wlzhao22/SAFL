# TODO: delete this file  
 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class IoUScoreLoss(nn.Module):
    def __init__(self, momentum=0.05, eps=1e-4) -> None:
        super().__init__()
        self.eps = eps 
        self.momentum = momentum 
        self.register_buffer('running_mean_1', torch.tensor(0.5, dtype=torch.float))
        self.register_buffer('running_std_1', torch.tensor(1, dtype=torch.float))
        self.register_buffer('running_mean_2', torch.tensor(0.5, dtype=torch.float))
        self.register_buffer('running_std_2', torch.tensor(1, dtype=torch.float))

    def forward(self, pred, target, n):
        if pred.numel() == 0: 
            return pred.new_tensor(0.)

        if pred.numel() > 1:
            pred_std = pred.std().detach()
            pred_mean = pred.mean().detach()
            target_std = target.std().detach()
            target_mean = target.mean().detach()

            self.running_std_1.mul_(1 - self.momentum)
            self.running_mean_1.mul_(1 - self.momentum)
            self.running_std_2.mul_(1 - self.momentum)
            self.running_mean_2.mul_(1 - self.momentum)

            self.running_std_1.add_(self.momentum * pred_std)
            self.running_mean_1.add_(self.momentum * pred_mean)
            self.running_std_2.add_(self.momentum * target_std)
            self.running_mean_2.add_(self.momentum * target_mean)

        pred_norm = (pred - self.running_mean_1) / self.running_std_1.clamp(min=self.eps)
        target_norm = (target - self.running_mean_2) / self.running_std_2.clamp(min=self.eps)
        return F.l1_loss(pred_norm, target_norm, reduction='none').sum() / n 
        