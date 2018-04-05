""" Implementation of multiple loss fuctions """
import torch.nn as nn
import torch

def pixel_loss(input, target, norm_constant):
	""" Pixel loss => normalized euclidean distance between the output image and the target """
	return torch.div(torch.sum((input - target) ** 2), norm_constant)

def perceptual_loss(input, target):
	""" Perceptual Loss, as proposed by Justin Johnson paper """
	pass