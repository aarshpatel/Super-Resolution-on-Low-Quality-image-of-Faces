from dataset import ObfuscatedDatasetLoader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from models.three_layer_cnn_baseline import ThreeLayerCNNBasline
from scripts.metrics import psnr, ssim

# get the arguments
num_epochs = 50
lr = .001
method = "blurred"
size = 4
batch_size = 8 
use_cuda = False

# setup the dataset
dset = ObfuscatedDatasetLoader("./data/lfw_preprocessed/cropped/", method, size)
train_loader = DataLoader(dset, batch_size=batch_size, shuffle=True)

# load in the appropriate model
cnn_baseline = ThreeLayerCNNBasline()

# setup the loss function (MSE or Perceptual Loss)
criterion = nn.MSELoss()

# setup the optimizer
optimizer = optim.Adam(cnn_baseline.parameters(),lr=lr)

# train the model
for epoch in range(num_epochs):

	epoch_loss = 0

	for iteration, batch in enumerate(train_loader, 1):
	    input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

	    # use the GPU
	    if use_cuda:
	        input = input.cuda()
	        target = target.cuda()

	    # zero out the gradients
	    optimizer.zero_grad() 

	    # compute output from cnn
	    model_out = cnn_baseline(input.float())

	    # compute the loss function 
	    loss = criterion(model_out, target.float())

	    # compute the epoch loss
	    epoch_loss += loss.data[0]

	    # backprop
	    loss.backward()


	    # perform a gradient step
	    optimizer.step()

	    print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(train_loader), loss.data[0]))

	print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_loader)))     



