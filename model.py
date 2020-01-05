import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, input_ch, output_ch):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

