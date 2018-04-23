import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class ThreeLayerCNNDisc(nn.Module):
    """ 
    Simple three layer baseline that uses padding to perseve the spatial size 
    This is similar to the SRCNN model that was presented here:  https://arxiv.org/pdf/1501.00092.pdf
    """

    def __init__(self, batchnorm=False):
        super(ThreeLayerCNNDisc,self).__init__()
        N = 3
        D_in = 5 
        D_mid = 5
        D_out = 3
        self.model = nn.Sequential(
            nn.Conv2d(N, D_in, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.BatchNorm2d(D_in),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(D_in, D_mid, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.BatchNorm2d(D_mid),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(D_mid, D_out,kernel_size=(2,2), stride=(2,2)),
            nn.Sigmoid()
        )

    def get_flat_fts(self, in_size, fts):
        f = fts(Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        out = self.model(x)
        out =self.classifier( out.view(-1, self.flat_fts))
        return out