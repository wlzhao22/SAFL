import torch
from torch import nn
from torch.nn import functional as F
from .DUpsample import DUpsample
from .shortcut_block import shortcut_block

class ERFNetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,  # [3, 32, 24, 40, 112, 320]
            decoder_channels,  # [128, 64, 16]
            shortcut_cfg=[2, 3, 4]
    ):
        super().__init__()

        assert len(decoder_channels) == len(shortcut_cfg)
        self.shortcut_cfg = shortcut_cfg

        # upsample layers
        self.upsample_layers = nn.ModuleList()
        self.upsample_layers.append(DUpsample(ninput=encoder_channels[-1], noutput=decoder_channels[0] * 4))
        self.upsample_layers.append(DUpsample(ninput=decoder_channels[0], noutput=decoder_channels[1] * 4))
        self.upsample_layers.append(DUpsample(ninput=decoder_channels[1], noutput=decoder_channels[2] * 4))

        # shortcut layer
        self.shortcut_layers = nn.ModuleList()
        for i in range(len(decoder_channels)):
            shortcut_input = encoder_channels[-i - 2]
            shortcut_output = decoder_channels[i]
            for j in range(shortcut_cfg[i]):
                self.shortcut_layers.append(shortcut_block(shortcut_input, shortcut_output, dropprob=0))
                shortcut_input = shortcut_output

        # upsample_2 layers
        self.upsample_2_layer = DUpsample(ninput=decoder_channels[-1], noutput=decoder_channels[-1])

    def forward(self, features):
        features = features[::-1]  # reverse channels to start from head of encoder
        output = features[0]
        shortcut_layer_num = 0
        outputs = []
        for i, upsample_layer in enumerate(self.upsample_layers):
            # upsample
            output = upsample_layer(output)
            output = torch.nn.PixelShuffle(2)(output)
            # shortcut
            shortcut = features[i + 1]
            for j in range(self.shortcut_cfg[i]):
                shortcut = self.shortcut_layers[shortcut_layer_num](shortcut)
                shortcut_layer_num += 1
            output = F.relu(output + shortcut)
            outputs.append(output)

        output = self.upsample_2_layer(output)
        output = torch.nn.PixelShuffle(2)(output)
        outputs.append(output)
        # print("features[i+1] ", features[i+1].shape)
        # print("shortcut ", shortcut.shape)
        # print("output ", output.shape)

        return outputs

