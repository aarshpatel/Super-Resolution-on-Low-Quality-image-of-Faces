import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from math import log10
import argparse
import numpy as np
import copy
from dataset import ObfuscatedDatasetLoader
from models.three_layer_cnn_baseline import ThreeLayerCNNBasline
from scripts.metrics import psnr, ssim
from scripts.plots import plot_training_loss, plot_train_val_psnr
from loss import pixel_loss

def train_model(model, input_size, loss, train_loader, val_loader, num_epochs, lr, model_hyperparameters):
	"""
	Train a 'facial reconstruction' model given the arguments

	model: pytorch model (ex. ThreeLayerCNNBaseline)
	input_size: size of the input image (eg. 110)
	train_loader: pytorch data loader for the training data
	val_loader: pytorch data loader for the validation data
	num_epochs: num of epochs to train the model
	lr: regularization of the optimizer, Adam
	model_hyperparameters: string representing the hyperparameters used training
	"""

	if loss == "pixel":
		criterion = nn.MSELoss()

	optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=.001)

	training_psnr = []
	validation_psnr = []
	training_loss_for_iterations = []

	best_val_psnr = 0.0
	best_model_weights = copy.deepcopy(model.state_dict())

	iterations = 0

	for epoch in range(num_epochs):

		epoch_loss = 0
		total_epoch_psnr = 0

		for iteration, batch in enumerate(train_loader, 1):
			iterations += 1
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
			# store the iteration loss after every 500 iterations
			if iterations % 500 == 0:
				training_loss_for_iterations.append((iterations, loss.data[0]))
			# aggregate the epoch loss
			epoch_loss += loss.data[0]
			# calculate the batch psnr
			psnr = 20 * log10(255/np.sqrt(loss.data[0]))
			total_epoch_psnr += psnr
			# backprop
			loss.backward()
			# perform a gradient step
			optimizer.step()

			if iterations % 500 == 0:
				print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iterations, len(train_loader), loss.data[0]))

		# compute the train for each batch
		avg_epoch_psnr = total_epoch_psnr / len(train_loader)
		training_psnr.append(avg_epoch_psnr)
		print("Epoch Training PSNR: ", avg_epoch_psnr)

		# compute the val for each batch
		avg_val_psnr = test_psnr(model, val_loader)
		validation_psnr.append(avg_val_psnr)
		print("Epoch Valiaation PSNR: ", avg_val_psnr)

		# if we find a model that does better in the val psnr then save its weights
		if avg_val_psnr > best_val_psnr:
			best_val_psnr = avg_val_psnr
			best_model_weights = copy.deepcopy(model.state_dict())

		print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_loader)))

	# plot the training loss and train/val psnr values
	plot_training_loss(training_loss_for_iterations, "figures/{0}_training_loss.png".format(model_hyperparameters))

	# plot the training psnr and validation psnr after every epoch
	plot_train_val_psnr(training_psnr, validation_psnr, "figures/{0}_training_validation_psnr.png".format(model_hyperparameters))

	model.load_state_dict(best_model_weights)
	return model

def test_psnr(model, data_loader):
	""" Calculate the avg psnr across the test set """
	avg_psnr = 0
	loss = nn.MSELoss()

	for batch in data_loader:
		input, target = Variable(batch[0]), Variable(batch[1])

		if use_cuda:
			input = input.cuda()
			target = target.cuda()

		prediction = model(input.float())

		mse = loss(prediction, target.float())
		psnr = 20 * log10(255/np.sqrt(mse.data[0]))
		avg_psnr += psnr

	return avg_psnr / float(len(data_loader))

def test_ssim(model):
	""" Calculate the avg. SSIM (Structural Similarity) across the test set """
	pass

def save_model(model, model_name, location):
	model_out_path = location + model_name + ".pth"
	torch.save(model.state_dict(), model_out_path)
	# torch.save(model, model_out_path)
	print("Model saved to {}".format(model_out_path))

if __name__ == "__main__":
    # get the arguments
    parser = argparse.ArgumentParser(description='Facial Reconstruction using CNNs')
    parser.add_argument("--model", type=str, default="ThreeLayerCNNBasline", help="type of model to use for facial reconstruction")
    parser.add_argument("--method", type=str, default="blurred", help="type of obfuscation method to use")
    parser.add_argument("--size", type=int, help="size of the obfuscation method applied to images")
    parser.add_argument("--loss", type=str, default="mse", help="type of loss function to use (eg. mse, perceptual)")
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=10, help='testing batch size')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    opt = parser.parse_args()

    num_epochs = opt.epochs
    lr = opt.lr
    method = opt.method
    size = opt.size
    batch_size = opt.batch_size
    use_cuda = opt.cuda
    loss = opt.loss
    num_workers = opt.threads

    main_hyperparameters = "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(opt.model, opt.method, opt.size, opt.loss, opt.lr, opt.epochs, opt.batch_size)

    print("Hyperparameters: ", main_hyperparameters)

    # get the training data
    train_dset = ObfuscatedDatasetLoader("./data/lfw_preprocessed/cropped_grayscale/", method, size, "train", train_mean=None, total_num_images=None)
    # train_mean = train_dset.train_mean
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # get the validation set
    val_dset = ObfuscatedDatasetLoader("./data/lfw_preprocessed/cropped_grayscale/", method, size, "val")
    val_loader = DataLoader(val_dset, shuffle=True, num_workers=num_workers)

    # get the test set
    test_dset = ObfuscatedDatasetLoader("./data/lfw_preprocessed/cropped_grayscale/", method, size, "test")
    test_loader = DataLoader(test_dset, shuffle=True, num_workers=num_workers)

    model = ThreeLayerCNNBasline()

    if use_cuda:
        model = model.cuda()

    input_size = 110
    trained_model = train_model(model, input_size, loss, train_loader, val_loader, num_epochs, lr, main_hyperparameters)

    # save the best model
    save_model(trained_model, main_hyperparameters, "saved_models/")

    avg_psnr_score_train = test_psnr(trained_model, train_loader)
    avg_psnr_score_test = test_psnr(trained_model, test_loader)

    print("AVG PSNR Score on train: ", avg_psnr_score_train)
    print("AVG PSNR Score on test: ", avg_psnr_score_test)

