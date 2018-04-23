""" Implementation of multiple loss fuctions """
import copy
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

def create_loss_model(vgg, end_layer, use_maxpool=True, use_cuda=False):
    """
        [1] uses the output of vgg16 relu2_2 layer as a loss function (layer8 on PyTorch default vgg16 model).
        This function expects a vgg16 model from PyTorch and will return a custom version up until layer = end_layer
        that will be used as our loss function.
    """

    vgg = copy.deepcopy(vgg)
    model = nn.Sequential()
    if use_cuda:
        model.cuda()

    i = 0
    for layer in list(vgg):
        if i > end_layer:
            break

        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            if use_maxpool:
                model.add_module(name, layer)
            else:
                avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
                model.add_module(name, avgpool)
        i += 1
    return model

def perceptual_loss(pred, target):
	vgg16 = models.vgg16(pretrained=True).features
	vgg16.cuda()

	# sequential model
	vgg_loss = create_loss_model(vgg16, 8, use_cuda=True)

	for param in vgg_loss.parameters():
		param.requires_grad = False

	inp = vgg_loss(pred.cuda())
	tar = vgg_loss(target.cuda())

	lossFunction = nn.MSELoss(size_average=False)
	loss = lossFunction(inp, tar)

	return loss