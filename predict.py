""" Predict images form the train and val set """
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
from dataset import ObfuscatedDatasetLoader
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from scripts.metrics import calc_psnr, calc_ssim
from scripts.average_meter import AverageMeter
import os
import torchvision.utils as vutils



def save_image(input, output, target, filename):
    """ Save the input, output, target image during training """
    all_images = torch.cat((input, output, target))
    vutils.save_image(all_images, filename=filename, normalize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prediction script to test performance of the model on the testing data')
    parser.add_argument("--model", type=str, help="the trained model to evaluate")
    parser.add_argument("--image", type=str, help="the image passed into the model")
    parser.add_argument('--cuda', action='store_true', help='use cuda?')

    opt = parser.parse_args()
    model_name = opt.model
    use_cuda = opt.cuda
    test_image = opt.image

    if os.path.isfile("/saved_models/" + str(model_name) + "/best_model.pth.tar"):
        print("=> loading checkpoint '{}'".format(model_name))
        test_model = torch.load("/saved_models/" + str(model_name) + "/best_model.pth.tar")
        test_model.cuda()
        test_model.eval()
        output = test_model(test_image)
        save_image(input=test_image, output=output, target=model_name, filename="sample_prediction_" + str(model_name) + "__" + str(test_image))
    else:
        print("=> no checkpoint found at '{}'".format("/saved_models/" + str(model_name) + "/best_model.pth.tar"))
