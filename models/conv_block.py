import torch
import torchvision
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride, pool=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                            stride, padding, bias=False)
        self.relu = nn.ReLU()
        self.use_pool = pool
        if pool:
            self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        if self.use_pool:
            out = self.pool(out)
        return out