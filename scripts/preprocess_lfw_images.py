""" Script to preprocess (crop, blur, pixelate) all images in the Labeled Faces in the Wild Dataset """

from PIL import Image, ImageFilter
import numpy as np
import os
import cv2 #opencv

# TODO
# 1) Make the code take in any dataset and apply any size crop, blurring, pixelation
# 2) Should be highly customizable

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

def grayscale(img):
    """ Grayscale an image """
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    return Image.fromarray(img_gray)

def create_train_test_val_sets(original_dataset_location):
    """ Load all images from the dataset and create train/val/test sets"""
    all_images = []
    file_names = []
    for subdir, _, files in os.walk(original_dataset_location):
        for file in files:
            image_file_path = os.path.join(subdir, file)
            if image_file_path.endswith(".jpg"):
                file_names.append(file)
                img = Image.open(image_file_path)
                img = crop_img_to_size(img, 110, None)
                img = grayscale(img)
                img = np.array(img)
                img = np.expand_dims(img, axis=0)
                all_images.append(img)

    print("Num of images in the dataset: ", len(all_images))
    all_images = np.array(all_images)
    print(all_images.shape)
    return
    train = all_images[:8000, :, :, :]
    val = all_images[8000:10646, :, :, :]
    test = all_images[10646:, :, :, :]

    print("Making the train/val/test sets...")
    num_img = 0
    print("Making train dataset ({0} num of images)".format(train.shape[0]))
    for row in train:
        img = Image.fromarray(row.reshape(110, 110))
        print("Saving file: ", file_names[num_img])
        img.save("../data/lfw_preprocessed/cropped/" + "/train/" + "{0}".format(file_names[num_img]))
        num_img += 1
    
    print("Making val dataset ({0} num of images)".format(val.shape[0]))
    for row in val:
        img = Image.fromarray(row.reshape(110, 110))
        img.save("../data/lfw_preprocessed/cropped/" + "/val/" + "{0}".format(file_names[num_img]))
        num_img += 1
    
    print("Making test dataset ({0} num of images)".format(test.shape[0]))
    for row in test:
        img = Image.fromarray(row.reshape(110, 110))
        img.save("../data/lfw_preprocessed/cropped/" + "/test/" + "{0}".format(file_names[num_img]))
        num_img += 1

if __name__ == '__main__':
    create_train_test_val_sets("../data/lfw/")
