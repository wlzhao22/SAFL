import torch
from torch import nn
from torch.nn import functional as F


class DUpsample (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        # self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.conv_1 = nn.Conv2d(ninput, noutput, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(noutput, eps=1e-3)
        self.conv_2 = nn.Conv2d(noutput, noutput, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv_1(input)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv_2(output)
        output = self.bn2(output)
        output = F.relu(output)
        return output