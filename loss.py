""" Implementation of multiple loss fuctions """
import torch.nn as nn

def pixel_loss(input, target, norm_constant):
	""" Pixel loss => normalized euclidean distance between the output image and the target """
	return torch.sum((input - target) ** 2) / norm_constant
