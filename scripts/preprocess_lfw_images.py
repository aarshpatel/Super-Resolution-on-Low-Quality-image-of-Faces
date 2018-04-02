""" Script to preprocess all images in the Labeled Faces in the Wild Dataset """

from PIL import Image, ImageFilter
import numpy as np
import cv2 #This will give an error if you don't have cv2 module
import os


def crop_img_to_size(img, size, output):
    """ Apply crop to image """
    width, height = img.size   # Get dimensions
    left = (width - size)/2
    top = (height - size)/2
    right = (width + size)/2
    bottom = (height + size)/2

    img = img.crop((left, top, right, bottom))
    return img

def apply_gaussian_blur(img, radius):
    """ Apply gaussian blur to an image """
    blur_image = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return blur_image

def pixelate(image, pixel_size):
    """ Pixelate an image """
    image = image.resize(
        (image.size[0] // pixel_size, image.size[1] // pixel_size),
        Image.NEAREST
    )
    image = image.resize(
        (image.size[0] * pixel_size, image.size[1] * pixel_size),
        Image.NEAREST
    )

    return image


print("Cropping all LFW images...")
for subdir, dirs, files in os.walk("../data/lfw/"):
    for file in files:
        image_file_path = os.path.join(subdir, file)
        if image_file_path.endswith(".jpg"):
            img = Image.open(image_file_path)
            crop_img = crop_img_to_size(img, 110, None)
            crop_img.save("../data/preprocessed_lfw/cropped/" + file)



output_dir = "../data/preprocessed_lfw/blurred/"
filters = [4, 6, 8]
print("Blurring all images in LFW...")
for idx, fn in enumerate(os.listdir("../data/preprocessed_lfw/cropped/")):
    image_file_path = "../data/preprocessed_lfw/cropped/" + fn
    if image_file_path.endswith(".jpg"):
        if idx % 1000 == 0:
            print("{0} images processed...".format(idx+1))
        img = Image.open(image_file_path)
        for filter_size in filters:
            blurred_image = apply_gaussian_blur(img, filter_size)
            blurred_image.save("../data/preprocessed_lfw/blurred/filter_" + str(filter_size) + "/" + fn)


output_dir = "../data/preprocessed_lfw/pixelated/"
sizes = [6,8,12]
print("Pixelating all images in LFW...")
for idx, fn in enumerate(os.listdir("../data/preprocessed_lfw/cropped/")):
    image_file_path = "../data/preprocessed_lfw/cropped/" + fn
    if image_file_path.endswith(".jpg"):
        if idx % 1000 == 0:
            print("{0} images processed...".format(idx))
        img = Image.open(image_file_path)
        for size in sizes:
            pixelated_image = pixelate(img, size)
            pixelated_image.save("../data/preprocessed_lfw/pixelated/size_" + str(size) + "/" + fn)