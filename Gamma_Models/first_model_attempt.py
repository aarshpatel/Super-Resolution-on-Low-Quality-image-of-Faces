from __future__ import print_function
import torch.nn as nn
import os
import numpy as np
from scipy import misc
import torch
from torch.autograd import Variable



def train(epoch, xdata, ydata):
    c = list(zip(xdata, ydata))
    np.random.shuffle(c)
    xdata, ydata = zip(*c)
    curr_loss = 0
    for file_index in range(1000,len(xdata)-5,5):
        data = Variable(torch.Tensor(np.array([np.array(misc.imread(fname)).T for fname in xdata[file_index:file_index+5]])))
        target = Variable(torch.Tensor(np.array([np.array(misc.imread(fname)).T for fname in ydata[file_index:file_index + 5]])))
        # print(file_index)
        # print(data.shape)
        # print(data)
        y_pred = model(data)
        loss = loss_fn(y_pred, target)
        model.zero_grad()
        loss.backward()
        curr_loss += loss.data[0]
        for param in model.parameters():
            param.data -= learning_rate * param.grad.data
        if file_index % 300 == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(epoch, file_index * len(data), len(xdata),100. * file_index / len(xdata), loss.data[0]))
            print("image number:" +str(file_index-2100) +", current loss: " + str(curr_loss/300))
            curr_loss = 0



def test():
    test_loss = 0
    for file_index in range(0,500):
        if file_index % 100 == 0:
            print("validating: " + str(file_index))
        data = Variable(torch.Tensor(np.array([np.array(misc.imread(fname)).T for fname in xdata[file_index:file_index + 5]])))
        target = Variable(torch.Tensor(np.array([np.array(misc.imread(fname)).T for fname in ydata[file_index:file_index + 5]])))
        y_pred = model(data)
        test_loss += loss_fn(y_pred, target)/500
    print("average loss: " + str(test_loss))



ydata = []
for dirpath, dirnames, filenames in os.walk("/Users/mike/Desktop/PycharmProjects/CS682-Project/data/preprocessed_lfw/cropped"):
    for filename in [f for f in filenames if f.endswith(".jpg")]:
        ydata.append(os.path.join(dirpath, filename))
print("imported cropped pictures to set y")


xdata = []
for dirpath, dirnames, filenames in os.walk("/Users/mike/Desktop/PycharmProjects/CS682-Project/data/preprocessed_lfw/blurred/filter_4"):
    for filename in [f for f in filenames if f.endswith(".jpg")]:
        xdata.append(os.path.join(dirpath, filename))
print("imported blurred (4) pictures to set x")



N = 3
D_in = 5
D_mid = 5
D_out = 3
model = nn.Sequential(
    nn.Conv2d(N, D_in, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
    nn.BatchNorm2d(D_in),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.BatchNorm2d(D_in),
    nn.Conv2d(D_in, D_mid, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
    nn.BatchNorm2d(D_mid),
    nn.ConvTranspose2d(D_mid, D_out,kernel_size=(2,2), stride=(2,2)),
)





loss_fn = nn.MSELoss()
learning_rate = 1e-4

for epoch in range(1, 10):
    train(epoch, xdata, ydata)
    test()