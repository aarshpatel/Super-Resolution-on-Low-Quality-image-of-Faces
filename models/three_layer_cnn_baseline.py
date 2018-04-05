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
        # self.conv_block_1 = ConvBlock(in_channel=1, out_channel=64, kernel_size=3, padding=0, stride=1, pool=False)
        # self.conv_block_2 = ConvBlock(in_channel=64, out_channel=32, kernel_size=3, padding=0, stride=1, pool=False)
        # self.conv_block_3 = ConvBlock(in_channel=32, out_channel=16, kernel_size=3, padding=0, stride=1, pool=False)
        # self.conv_block_4 = ConvBlock(in_channel=16, out_channel=8, kernel_size=3, padding=0, stride=1, pool=False)
        # self.conv_block_5 = ConvBlock(in_channel=8, out_channel=1, kernel_size=1, padding=0, stride=1, pool=False)
        # self.upsampling = nn.modules.UpsamplingBilinear2d(size=(110, 110))

        N = 1
        D_in = 5 
        D_mid = 5
        D_out = 3

        self.model = nn.Sequential(
            nn.Conv2d(N, D_in, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.BatchNorm2d(D_in),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(D_in),
            nn.Conv2d(D_in, D_mid, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.BatchNorm2d(D_mid),
            nn.ConvTranspose2d(D_mid, D_out,kernel_size=(2,2), stride=(2,2)),
        )

    def forward(self,x):
        # out = self.conv_block_1(x)
        # out = self.conv_block_2(out)
        # out = self.conv_block_3(out)
        # out = self.conv_block_4(out)
        # out = self.conv_block_5(out)
        # out = self.upsampling(out)
        out = self.model(x)
        return out
