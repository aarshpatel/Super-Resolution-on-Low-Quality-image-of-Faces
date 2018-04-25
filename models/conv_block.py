import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Simple ConvBlock (Conv2d -> BatchNorm -> Relu)
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.cnn(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        return out