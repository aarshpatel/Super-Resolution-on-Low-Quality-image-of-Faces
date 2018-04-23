import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from math import log10
import argparse
import numpy as np
import copy
import time
import os
import shutil
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from dataset import ObfuscatedDatasetLoader
from models.three_layer_cnn_baseline import ThreeLayerCNNBasline
from scripts.metrics import psnr, ssim
from scripts.plots import plot_training_loss, plot_train_val_psnr
from loss import pixel_loss


def train(train_loader, model, criterion, optimizer, epoch):
	""" Train the model for one epoch """

	batch_time_meter = AverageMeter()
	losses_meter = AverageMeter()
	psnr_meter = AverageMeter()

	# set the model to train mode
	model.train()
	
	start = time.time()
	for iteration, batch in enumerate(train_loader, 1):
			input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

			# use the GPU
			if use_cuda:
			    input = input.cuda()
			    target = target.cuda()

			# compute output from CNN model
			output = model(input.float())
			loss = criterion(output, target.float())

			# measure psnr and loss
			psnr = calc_psnr(loss.data[0])
			psnr_meter.update(psnr, input.size(0))
			losses_meter.update(loss.data[0], input.size(0))

			# zero out the gradients
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# measure the time it takes to train for one epoch
			batch_time_meter.update(time.time() - start)			
			start = time.time()

			if iteration % 100 == 0:

				model_output_image = output.data
				model_input_image = input.data.float()
				model_target_image = target.data.float()

				# vis_image = torch.cat((model_output_image, model_input_image, model_target_image))
				vis_image = torch.cat((model_input_image, model_target_image))

				if opt.tensorboard:

					x = vutils.make_grid(vis_image, normalize=True)
					y = vutils.make_grid(model_output_image, normalize=True)

					writer.add_image('Input-Image', x, epoch*iteration)
					writer.add_image('Reconsructed-Image', y, epoch*iteration)
					
				if opt.save_img:					
					save_img_output_filename = "./saved_image_from_runs/{0}_epoch_{1}_iter_output.jpg".format(epoch, iteration)
					save_img_input_target_filename = "./saved_image_from_runs/{0}_epoch_{1}_iter_gt.jpg".format(epoch, iteration)
					vutils.save_image(model_output_image, filename=save_img_output_filename, normalize=True)
					vutils.save_image(vis_image, filename=save_img_input_target_filename, normalize=True)
					
					
				print('Epoch: [{0}][{1}/{2}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'PSNR {psnr.val:.3f} ({psnr.avg:.3f})'.format(
						epoch, iteration, len(train_loader), batch_time=batch_time_meter,
						loss=losses_meter, psnr=psnr_meter))

	# log value to tensorboard or visdom
	if opt.tensorboard:
		writer.add_scalar("PSNR/Train", psnr_meter.avg, epoch)
		writer.add_scalar("Loss/Train", losses_meter.avg, epoch)

def validate(val_loader, model, criterion, epoch):
	""" Validate the model on the validation set """
	batch_time_meter = AverageMeter()
	losses_meter = AverageMeter()
	psnr_meter = AverageMeter()

	# switch to eval mode
	model.eval()

	start = time.time()

	for iteration, batch in enumerate(val_loader, start=1):
		input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

		# use the GPU
		if use_cuda:
			input = input.cuda()
			target = target.cuda()

		# compute output from CNN model
		output = model(input.float())
		loss = criterion(output, target.float())

		# compute the psnr and loss on the validation set
		psnr = calc_psnr(loss.data[0])
		psnr_meter.update(psnr, input.size(0))
		losses_meter.update(loss.data[0], input.size(0))

		# measure time
		batch_time_meter.update(time.time() - start)			
		start = time.time()

		if iteration % 100 == 0:
			print('Test: [{0}/{1}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'PSNR {psnr.val:.3f} ({psnr.avg:.3f})'.format(
						iteration, len(val_loader), batch_time=batch_time_meter, loss=losses_meter,
						psnr=psnr_meter))

	print("AVG PSNR after epoch {0}: {1}".format(epoch, psnr_meter.avg))

	if opt.tensorboard:
		writer.add_scalar("PSNR/Val", psnr_meter.avg, epoch)
		writer.add_scalar("Loss/Val", losses_meter.avg, epoch)		

	return losses_meter.avg, psnr_meter.avg

def calc_psnr(mse):
	"""Calculate the psnr (Peak Signal Noise Ratio)"""
	return 20 * log10(255.0/np.sqrt(mse))	

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(name, state, is_best, filename='checkpoint.pth.tar'):
    """Saves model checkpoint to disk"""
    directory = "model_runs/%s/"%(name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_runs/%s/'%(name) + 'model_best.pth.tar')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Facial Reconstruction using CNNs')
	parser.add_argument("--model", type=str, default="ThreeLayerCNNBasline", help="type of model to use for facial reconstruction")
	parser.add_argument("--method", type=str, default="blurred", help="type of obfuscation method to use")
	parser.add_argument("--size", type=int, help="size of the obfuscation method applied to images")
	parser.add_argument('--grayscale', action="store_true", help="use grayscale images?")
	parser.add_argument("--loss", type=str, default="mse", help="type of loss function to use (eg. mse, perceptual)")
	parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
	parser.add_argument('--test_batch_size', type=int, default=10, help='testing batch size')
	parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train for')
	parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
	parser.add_argument('--weight-decay', type=float, default=1e-4, help="weight decay applied to the optimizer")
	parser.add_argument('--cuda', action='store_true', help='use cuda?')
	parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
	parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
	parser.add_argument('--tensorboard', action="store_true", help="use tensorboard for visualization?")
	parser.add_argument('--save_img', action="store_true", help="save the output images when training the model")
	global opt, writer, best_avg_psnr
	opt = parser.parse_args()

	# setup the tensorboard 
	writer = SummaryWriter("./runs/")

	best_avg_psnr = 0

	# get the arguments from argparse
	num_epochs = opt.epochs
	lr = opt.lr
	method = opt.method
	size = opt.size
	batch_size = opt.batch_size
	use_cuda = opt.cuda
	loss = opt.loss
	num_workers = opt.threads
	weight_decay = opt.weight_decay
	grayscale = opt.grayscale

	main_hyperparameters = "{0}_method={1}_size={2}_loss={3}_lr={4}_epochs={5}_batch_size={6}".format(opt.model, opt.method, opt.size, opt.loss, opt.lr, opt.epochs, opt.batch_size)

	print("Hyperparameters: ", main_hyperparameters)

	if grayscale:
		image_color = "grayscale"
	else:
		image_color = "rgb"

	#################
	# Normalization # 
	#################
	train_mean = np.array([150.79660111, 115.31313646,  94.28781092])
	train_std = np.array([52.17929494, 44.20110692, 42.75483222])
	normalize = transforms.Normalize(mean=[x/255.0 for x in train_mean],
                                     std=[x/255.0 for x in train_std])

	transform_normalize = transforms.Compose([
		transforms.ToTensor(),
		normalize,
    ])

	# get the training data
	train_dset = ObfuscatedDatasetLoader("./data/lfw_preprocessed/cropped_{}/".format(image_color), method, size, grayscale=False, data_type="train", transform=transform_normalize)
	train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	# get the validation set
	val_dset = ObfuscatedDatasetLoader("./data/lfw_preprocessed/cropped_{}/".format(image_color), method, size, grayscale=False, data_type="val", transform=transform_normalize)
	val_loader = DataLoader(val_dset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

	# get the test set
	# test_dset = ObfuscatedDatasetLoader("./data/lfw_preprocessed/cropped_grayscale/", method, size, "test")
	# test_loader = DataLoader(test_dset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

	# get the model
	model = ThreeLayerCNNBasline()

	# get the loss function
	if loss == "pixel":
		if use_cuda: criterion = nn.MSELoss().cuda()
		else:        criterion = nn.MSELoss()
	elif loss == "perceptual":
		print("loading the perceptual loss")

	# set the optimizer
	optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)

	# set the scheduler
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

	if use_cuda:
		model = model.cuda()

	for epoch in range(num_epochs):

		# trains the model for one epoch
		train(train_loader, model, criterion, optimizer, epoch)

		# evaluate on the validation set	
		val_loss, val_psnr_avg = validate(val_loader, model, criterion, epoch)

		# adjust the learning rate if val loss stops improving
		scheduler.step(val_loss)

		# remember the best psnr value and save the checkpoint model
		is_best = val_psnr_avg > best_avg_psnr 
		best_avg_psnr = max(val_psnr_avg, best_avg_psnr)
		save_checkpoint(main_hyperparameters,{
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_psnr': best_avg_psnr,
		}, is_best)

	print("Best PSNR on the validation set: {}".format(best_avg_psnr))