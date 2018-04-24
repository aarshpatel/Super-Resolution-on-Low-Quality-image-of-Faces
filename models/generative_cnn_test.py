import torch.nn as nn

class ThreeLayerCNNGen(nn.Module):
    """
    Simple three layer baseline that uses Conv Tranpose to upscale the image to the input image size
    """

    def __init__(self):
        super(ThreeLayerCNNGen,self).__init__()
        N = 3
        D_in = 5

        self.L1 = nn.Conv2d(N, D_in, kernel_size=(3,3), stride=(1,1), padding=(2,2))
        self.L2 = nn.PReLU()
        self.L3 = nn.Conv2d(D_in, D_in, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
        self.L4 = nn.BatchNorm2d(D_in)

    def forward(self,x):
        out = self.L2(self.L1(x))
        for i in range(4):
            out = ResidualBlock.forward(out)
        out = self.L4(self.L3(out))
        return out


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self):
        super(ResidualBlock, self).__init__()
        N = 3
        D_in = 5
        D_out = 5
        self.layer1 = nn.Conv2d(N, D_in, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
        self.layer2 = nn.BatchNorm2d(D_in)
        self.layer3 = nn.PReLU()
        self.layer4 = nn.Conv2d(N, D_in, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
        self.layer5 = nn.BatchNorm2d(D_in)

    def forward(self, x):
        residual = x

        out = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        out += residual
        return out


