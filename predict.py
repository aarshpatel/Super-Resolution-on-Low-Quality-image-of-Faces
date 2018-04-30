""" Predict images form the train and val set """
import torch
import argparse
import os
import torchvision.utils as vutils
from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable


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
    clean = Image.open(test_image)
    blurred = apply_gaussian_blur(clean, radius=4)
    if os.path.isfile("saved_models/" + str(model_name) + "model_best.pth.tar"):
        print("=> loading checkpoint '{}'".format(model_name))
        model = torch.load("saved_models/" + str(model_name) + "model_best.pth.tar")
        model.cuda()

        # convert to pytorch tensor

        # convert variable

        # convert to cuda
        train_mean = np.array([149.59638197, 114.21029544, 93.41318133])
        train_std = np.array([52.54902009, 44.34252746, 42.88273568])
        normalize = transforms.Normalize(mean=[mean / 255.0 for mean in train_mean],
                                         std=[std / 255.0 for std in train_std])

        transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        # normalize train_mean and train_std
        blurred = np.array(blurred)
        blurred = blurred.astype(float)
        blurred = torch.from_numpy(blurred)
        blurred = Variable(normalize(blurred)).cuda()
        test_image = np.array(clean)
        test_image = test_image.astype(float)
        test_image = torch.from_numpy(test_image)
        test_image = Variable(normalize(test_image)).cuda()
        output = model(blurred)
        save_image(input=blurred, output=output, target=model_name, filename=str(model_name) + "_Prediction.jpg")
        save_image(input=blurred, output=output, target=model_name, filename=str(model_name) + "_Ground_truth.jpg")
        save_image(input=blurred, output=output, target=model_name, filename=str(model_name) + "_Blurred.jpg")
    else:
        print("=> no checkpoint found at '{}'".format("saved_models/" + str(model_name) + "best_model.pth.tar"))
