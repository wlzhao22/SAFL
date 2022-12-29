import torch
from torch import nn
from torch.nn import functional as F


class shortcut_block (nn.Module):
    def __init__(self, inchann, outchann, dropprob):
        super().__init__()

        self.conv1x1 = nn.Conv2d(inchann, outchann, 1, stride=1, padding=0, bias=True)
        self.conv3x1 = nn.Conv2d(outchann, outchann, (3, 1), stride=1, padding=(1 ,0), bias=True)
        self.conv1x3 = nn.Conv2d(outchann, outchann, (1, 3), stride=1, padding=(0 ,1), bias=True)
        self.bn1 = nn.BatchNorm2d(outchann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(outchann, outchann, (3, 1), stride=1, padding=(1 ,0), bias=True, dilation = (1 ,1))
        self.conv1x3_2 = nn.Conv2d(outchann, outchann, (1 ,3), stride=1, padding=(0 ,1), bias=True, dilation = (1, 1))
        self.bn2 = nn.BatchNorm2d(outchann, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv1x1(input)
        output = F.relu(output)

        output = self.conv3x1(output)
        output = F.relu(output)
        output = self.conv1x3(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        output = F.relu(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return output