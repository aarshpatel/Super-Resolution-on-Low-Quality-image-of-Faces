import torch
import torchvision
import torch.nn as nn

class ThreeLayerCNNBasline(nn.Module):
    """ Simple three layer baseline that uses padding to perseve the spatial size """
    def __init__(self, batchnorm=False):

        super(ThreeLayerCNNBasline,self).__init__()
        self.conv1 = nn.Conv2d(1,128,kernel_size=9);
        self.relu1 = nn.ReLU();
        self.conv2 = nn.Conv2d(128,64,kernel_size=3);
        self.relu2 = nn.ReLU();
        self.conv3 = nn.Conv2d(64,1,kernel_size=5);
        
        # add bilinear upsampling
        self.upsample = nn.modules.Upsample(size=(110, 110), mode="bilinear")

    def forward(self,x):

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.upsample(out)

        return out
