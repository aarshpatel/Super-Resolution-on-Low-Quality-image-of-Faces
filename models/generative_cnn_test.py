import torch.nn as nn

class ThreeLayerCNNGen(nn.Module):
    """
    Simple three layer baseline that uses Conv Tranpose to upscale the image to the input image size
    """

    def __init__(self):
        super(ThreeLayerCNNGen,self).__init__()
        N = 3
        D_in = 5

        #Before First residual Block
        self.block0 = nn.Sequential(
            nn.Conv2d(N, 64, kernel_size=4, stride=2, padding=1),
            nn.PReLU()
        )
        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64)
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 12, kernel_size=3, stride=1, padding=2),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.block8 = nn.Sequential(
            nn.Conv2d(N, N, kernel_size=3, stride=1, padding=2),
            nn.PixelShuffle(1),
            nn.PReLU()
        )
        self.out = nn.Conv2d(N, N, kernel_size=4, stride=2, padding=1),
    def forward(self, x):
        block0 = self.block0(x)
        block1 = self.block1(block0)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = block5 + self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7)
        out = self.out(block8)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, N):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Conv2d(N, N, kernel_size=3, stride=1, padding=2)
        self.layer2 = nn.BatchNorm2d(N)
        self.layer3 = nn.PReLU()
        self.layer4 = nn.Conv2d(N, N, kernel_size=3, stride=1, padding=2)
        self.layer5 = nn.BatchNorm2d(N)

    def forward(self, x):
        residual = x
        out = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        out += residual
        return out

