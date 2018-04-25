import torch.nn as nn
from conv_block import ConvBlock

class ThreeLayerCNNBaseline(nn.Module):
    """ 
    Simple three layer baseline that uses Conv Tranpose to upscale the image to the input image size 
    """

    def __init__(self, num_convblocks=1):
        super(ThreeLayerCNNBaseline,self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=1), 
            nn.MaxPool2d(2)
        )

        inner_cnn_blocks = [ConvBlock(64, 64) for _ in range(num_convblocks)]
        self.blocks = nn.Sequential(*inner_cnn_blocks)

        self.upsample = nn.ConvTranspose2d(64, 3,kernel_size=6, stride=2, padding=1)

    def forward(self,x):
        # input => 3 * 110 * 110
        out = self.layer_1(x) # 64 * 54 * 54
        out = self.blocks(out) # 64 * 54 * 54
        out = self.upsample(out) # 3 * 110 * 110
        return out


