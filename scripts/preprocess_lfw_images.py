""" Script to preprocess (crop, blur, pixelate) all images in the Labeled Faces in the Wild Dataset """

from PIL import Image, ImageFilter
import numpy as np
import os
import cv2 #opencv
import argparse

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

def create_train_test_val_sets(input_dir, output_dir, crop_size=110, grayscale=True):
    """ Load all images from the dataset and create train/val/test sets"""

    all_images = []
    file_names = []

    if grayscale: image_type = "grayscale"
    else: image_type = "rgb"

    new_output_dir = output_dir + "cropped_{}/".format(image_type)

    if not os.path.exists(new_output_dir):
        print("Making dir: ", new_output_dir)
        os.makedirs(new_output_dir)


    for subdir, _, files in os.walk(input_dir):
        for file in files:
            image_file_path = os.path.join(subdir, file)
            if image_file_path.endswith(".jpg"):
                file_names.append(file)
                img = Image.open(image_file_path)
                img = crop_img_to_size(img, crop_size, None)

                if grayscale:
                    img = grayscale(img)

                img = np.array(img)
                img = np.expand_dims(img, axis=0)
                all_images.append(img)

    print("Num of images in the dataset: ", len(all_images))

    all_images = np.array(all_images)
    train = all_images[:8000, :, :, :]
    val = all_images[8000:10646, :, :, :]
    test = all_images[10646:, :, :, :]

    # Make the train/val/test folders
    for data_type in ["train", "val", "test"]:
        if not os.path.exists(new_output_dir + data_type + "/"):
            print("Making dir: {0}".format(new_output_dir + data_type + "/"))
            os.makedirs(new_output_dir + data_type + "/")

    num_img = 0
    print("Making train dataset ({0} num of images)".format(train.shape[0]))
    for row in train:
        if grayscale:
            img = Image.fromarray(row.reshape(crop_size, crop_size))
        else:
            img = Image.fromarray(row.reshape(crop_size, crop_size, 3))
        print("Saving file: ", file_names[num_img])
        img.save("../data/lfw_preprocessed/cropped_" + image_type + "/train/" + "{}".format(file_names[num_img]))
        num_img += 1
    
    print("Making val dataset ({0} num of images)".format(val.shape[0]))
    for row in val:
        if grayscale:
            img = Image.fromarray(row.reshape(crop_size, crop_size))
        else:
            img = Image.fromarray(row.reshape(crop_size, crop_size, 3)) 
        img.save("../data/lfw_preprocessed/cropped_" + image_type + "/val/" + "{}".format(file_names[num_img]))
        num_img += 1
    
    print("Making test dataset ({0} num of images)".format(test.shape[0]))
    for row in test:
        if grayscale:
            img = Image.fromarray(row.reshape(crop_size, crop_size))
        else:
            img = Image.fromarray(row.reshape(crop_size, crop_size, 3))
        img.save("../data/lfw_preprocessed/cropped_" + image_type + "/test/" + "{}".format(file_names[num_img]))
        num_img += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprecessing of images')
    parser.add_argument("--input_dir", type=str, help="location of where original images are")
    parser.add_argument("--output_dir", type=str, help="location of where to put the preprocessed images")
    parser.add_argument("--crop_size", type=int, default=110, help="size of the cropped image")
    parser.add_argument("--grayscale", action='store_true', help="convert images to grayscale?")
    opt = parser.parse_args()
    
    input_dir = opt.input_dir 
    output_dir = opt.output_dir
    crop_size = opt.crop_size
    grayscale = opt.grayscale

    create_train_test_val_sets(input_dir=input_dir, output_dir=output_dir, crop_size=crop_size, grayscale=grayscale)
