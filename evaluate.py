""" Script to evaluate a model performance on the test data """
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
from dataset import ObfuscatedDatasetLoader
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from scripts.metrics import calc_psnr, calc_ssim
from models.three_layer_cnn_baseline import ThreeLayerCNNBaseline

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
	
def evaluate_on_test(model, test_loader, metric):
	""" Evaluate the trained model on the test data using some specified metric """

	metric_meter = AverageMeter()

	# switch to eval mode
	model.eval()

	loss_fn = nn.MSELoss().cuda()

	for _, batch in enumerate(test_loader, start=1):
		input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

		# use the GPU
		if use_cuda:
			input = input.cuda()
			target = target.cuda()

		# compute output from CNN model
		output = model(input)

		# compute the psnr and ssim on the test set
		if "psnr" == metric:
			mse = loss_fn(output, target)
			psnr = calc_psnr(mse.data[0])
			metric_meter.update(psnr, input.size(0))
		elif "ssim" == metric:
			ssim = calc_ssim(output.data.float().cpu().numpy(), target.data.float().cpu().numpy())
			metric_meter.update(ssim, input.size(0))

	return metric_meter.avg 

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Evaluation script to test performance of the model on the testing data')
	parser.add_argument("--model", type=str, help="the trained model to evaluate")
	parser.add_argument('--grayscale', action="store_true", help="use grayscale images?")
	parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
	parser.add_argument('--metrics', nargs='+', help="which metrics to evaluate on")
	parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
	parser.add_argument('--cuda', action='store_true', help='use cuda?')
		
	opt = parser.parse_args()
	model_name = opt.model


	grayscale = opt.grayscale
	batch_size = opt.batch_size
	metrics = opt.metrics
	num_workers = opt.threads
	use_cuda = opt.cuda
	
	model_name_split_parameters = model_name.split("/")[2].split("_")
	method = model_name_split_parameters[1].split("=")[1]
	size = int(model_name_split_parameters[2].split("=")[1])
	model = model_name_split_parameters[0]

	if grayscale:
		image_color = "grayscale"
	else:
		image_color = "rgb"	

	train_mean = np.array([149.59638197, 114.21029544,  93.41318133])
	train_std = np.array([52.54902009, 44.34252746, 42.88273568])
	normalize = transforms.Normalize(mean=[mean/255.0 for mean in train_mean],
										std=[std/255.0 for std in train_std])
	transform_normalize = transforms.Compose([
		transforms.ToTensor(),
		normalize,
	])

	# get the testing data
	test_dset = ObfuscatedDatasetLoader("./data/lfw_preprocessed/cropped_{}/".format(image_color), method, size,
											grayscale=False, data_type="test", transform=transform_normalize)
	test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	if model == "ThreeLayerCNNBaseline":
		print("Loading model: ", model)
		model = ThreeLayerCNNBaseline()
	else:
		print("Loading some other model")	


	trained_model_params = torch.load(model_name)

	# load in the model weights
	model.load_state_dict(trained_model_params['state_dict'])

	model.cuda()

	# evaluate the model on the test data
	for metric in metrics:

		metric_avg = evaluate_on_test(model, test_loader, metric)

		if metric == "psnr":
			print "PSNR Avg: ", metric_avg
		elif metric == "ssim":
			print "SSIM Avg: ", metric_avg