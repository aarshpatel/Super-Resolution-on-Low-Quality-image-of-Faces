from dataset import ObfuscatedDatasetLoader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from math import log10

from models.three_layer_cnn_baseline import ThreeLayerCNNBasline
from scripts.metrics import psnr, ssim

def train_model(model, loss, train_loader, num_epochs):
	""" 
	Train a 'facial reconstruction' model given the arguments 

	model: pytorch model (ex. ThreeLayerCNNBaseline)
	loss: type of loss function (MSE, Perceptual Loss)
	train_loader: pytorch data loader for the training data
	num_epochs: num of epochs to train the model
	lr: regularization of the optimizer, Adam
	"""

	optimizer = optim.Adam(model.parameters(),lr=lr)

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
		    model_out = model(input.float())
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


def test_psnr(model, test_loader):
	""" Calculate the avg psnr across the test set """
	avg_psnr = 0
	loss = nn.MSELoss()

	for batch in test_loader:
		input, target = Variable(batch[0]), Variable(batch[1])

		if use_cuda:
			input = input.cuda()
			target = target.cuda()

		prediction = model(input)
		mse = loss(prediction, target) 
		psnr = 10 * log10(1/mse.data[0])
		avg_psnr += psnr
	return avg_psnr / float(len(test_loader))

def test_ssim(model):
	""" Calculate the avg. SSIM (Structural Similarity) across the test set """
	pass


if __name__ == "__main__":
	# get the arguments
	num_epochs = 50
	lr = .001
	method = "blurred"
	size = 4
	batch_size = 8 
	use_cuda = False

	# get the training data 
	train_dset = ObfuscatedDatasetLoader("./data/lfw_preprocessed/cropped/", method, size, "train")
	train_loader = DataLoader(train_dset, batch_size=1, shuffle=True)

	# get the validation set
	val_dset = ObfuscatedDatasetLoader("./data/lfw_preprocessed/cropped/", method, size, "val")
	val_loader = DataLoader(val_dset, shuffle=True)

	# get the test set
	test_dset = ObfuscatedDatasetLoader("./data/lfw_preprocessed/cropped/", method, size, "test")
	test_loader = DataLoader(test_dset, shuffle=True)

	model = ThreeLayerCNNBasline()

	if use_cuda:
		model = model.cuda()

	loss = nn.MSELoss()

	train_model(model, loss, train_loader, num_epochs)
