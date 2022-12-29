import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

# class FocalLoss2d(nn.Module):

#     def __init__(self, gamma=0.25, weight=None, size_average=True):
#         super(FocalLoss2d, self).__init__()

#         self.gamma = gamma
#         self.weight = weight
#         self.size_average = size_average

#     def forward(self, input, target):
#         # if input.dim()>2:
#         #     print(f"====1========={input.shape}==========")
#         #     input = input.contiguous().view(input.size(0), input.size(1), -1)
#         #     print(f"====2========={input.shape}==========")
#         #     input = input.transpose(1,2)
#         #     print(f"====3========={input.shape}==========")
#         #     input = input.contiguous().view(-1, input.size(2)).squeeze()
#         #     print(f"====4========={input.shape}==========")
#         # if target.dim()==4:
#         #     target = target.contiguous().view(target.size(0), target.size(1), -1)
#         #     target = target.transpose(1,2)
#         #     target = target.contiguous().view(-1, target.size(2)).squeeze()
#         # else:
#         #     target = target.view(-1)

#         # compute the negative likelyhood
#         weight = Variable(self.weight)
#         logpt = -F.cross_entropy(input, target, ignore_index=-1)
#         pt = torch.exp(logpt)

#         # compute the loss
#         loss = -((1-pt)**self.gamma) * logpt

#         # averaging (or not) loss
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()

# class FocalLoss2d(nn.Module):

#     def __init__(self, gamma=2.0, weight=None, size_average=True):
#         super(FocalLoss2d, self).__init__()

#         self.gamma = gamma
#         self.weight = weight
#         self.size_average = size_average
#         self.nll_loss = torch.nn.NLLLoss2d(weight, size_average)

#     def forward(self, input, target):
#         return self.nll_loss((1 - F.softmax(input,1)) ** self.gamma * F.log_softmax(input,1), target)

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='softmax'):
        if type=='sigmoid':
            target = target.view(-1, 1).long()
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]
            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)
            class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
            class_weight = torch.gather(class_weight, 0, target)
            prob       = (prob*select).sum(1).view(-1,1)
            prob       = torch.clamp(prob,1e-8,1-1e-8)
            batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()
            if self.size_average:
                loss = batch_loss.mean()
            else:
                loss = batch_loss
        elif  type=='softmax':
            self.nll_loss = torch.nn.NLLLoss2d(class_weight, self.size_average)
            loss = self.nll_loss((1 - F.softmax(logit,1)) ** self.gamma * F.log_softmax(logit,1), target)
        
        return loss
