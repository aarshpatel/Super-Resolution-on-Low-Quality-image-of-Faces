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
from PIL import Image, ImageFilter

def save_image(input, output, target, filename):
    """ Save the input, output, target image during training """
    all_images = torch.cat((input, output, target))
    vutils.save_image(all_images, filename=filename, normalize=True)

def apply_gaussian_blur(img, radius):
    """ Apply gaussian blur to an image """
    blur_image = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return blur_image

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
    directory = os.listdir("data/lfw_preprocessed/cropped_rgb/test/")
    clean = Image.open(os.path.join(directory,test_image))
    blurred = apply_gaussian_blur(clean, radius=4)
    if os.path.isfile("saved_models/" + str(model_name) + "best_model.pth.tar"):
        print("=> loading checkpoint '{}'".format(model_name))
        test_model = torch.load("saved_models/" + str(model_name) + "best_model.pth.tar")
        test_model.cuda()
        test_model.eval()
        output = test_model(blurred)
        save_image(input=blurred, output=output, target=model_name, filename=str(model_name) + "_Prediction__" + str(output))
        save_image(input=blurred, output=output, target=model_name, filename=str(model_name) + "_Ground_truth__" + str(test_image))
        save_image(input=blurred, output=output, target=model_name, filename=str(model_name) + "_Blurred__" + str(blurred))
    else:
        print("=> no checkpoint found at '{}'".format("saved_models/" + str(model_name) + "best_model.pth.tar"))
