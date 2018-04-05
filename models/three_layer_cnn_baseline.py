import torch
import torchvision
import torch.nn as nn
from conv_block import ConvBlock 

import torch
import torchvision
import torch.nn as nn


class ThreeLayerCNNBasline(nn.Module):
    """ 
    Simple three layer baseline that uses padding to perseve the spatial size 
    This is similar to the SRCNN model that was presented here:  https://arxiv.org/pdf/1501.00092.pdf
    """

    def __init__(self, batchnorm=False):

        super(ThreeLayerCNNBasline,self).__init__()
        self.conv_block_1 = ConvBlock(in_channel=1, out_channel=64, kernel_size=3, padding=0, stride=1)
        self.conv_block_2 = ConvBlock(in_channel=64, out_channel=128, kernel_size=3, padding=0, stride=1, pool=False)
        self.conv_block_3 = ConvBlock(in_channel=128, out_channel=256, kernel_size=3, padding=0, stride=1, pool=False)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, padding=0, stride=1)
        self.relu1 = nn.ReLU()
        self.upsampling = nn.modules.UpsamplingBilinear2d(size=(110, 110))
        self.sigmoid_out = nn.Sigmoid()

    def forward(self,x):
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.upsampling(out)
        out = self.sigmoid_out(out)
        return out

