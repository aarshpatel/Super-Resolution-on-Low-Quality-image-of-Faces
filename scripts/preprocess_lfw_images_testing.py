""" Script to preprocess all images in the Labeled Faces in the Wild Dataset """

from PIL import Image, ImageFilter
import numpy as np
import cv2 
import os

def apply_bifilt_blur(img, radius):
    """ Apply bilateral blur to an image """
    blur_image = cv2.bilateralFilter(img, radius, 75, 75)
    return blur_image
def apply_avg_blur(img, radius):
    """ Apply averaged blur to an image """
    blur_image = cv2.blur(img, (radius, radius))
    return blur_image
def apply_med_blur(img, radius):
    """ Apply median blur to an image """
    blur_image = cv2.medianBlur(img, radius+1)
    return blur_image
def apply_rand_black(img, radius):
    """ Apply random black pixels to an image """
    image = cv2.imread(img)
    mask = np.zeros(image.shape)
    value = np.random.randint(0,100,(mask.shape[0],mask.shape[1]))*radius
    for row in range(mask.shape[0]):
        for pixel in range(mask.shape[1]):
            if value[row][pixel] > 10:
                mask[row][pixel] = image[row][pixel]
    return mask

def bifilt_blur_images(lfw_location, output_location, filter_sizes):
    print("Bifilter Blurring all images in LFW...")
    for idx, fn in enumerate(os.listdir(lfw_location)):
        image_file_path = lfw_location + fn
        if image_file_path.endswith(".jpg"):
            if idx % 1000 == 0:
                print("{0} images processed...".format(idx+1))
            img = cv2.imread(image_file_path)
            for filter_size in filter_sizes:
                blurred_image = apply_bifilt_blur(img, filter_size)

                cv2.imwrite(output_location + "filter_" + str(filter_size) + "/" + fn, blurred_image)
def avg_blur_images(lfw_location, output_location, filter_sizes):
    print("Average Blurring all images in LFW...")
    for idx, fn in enumerate(os.listdir(lfw_location)):
        image_file_path = lfw_location + fn
        if image_file_path.endswith(".jpg"):
            if idx % 1000 == 0:
                print("{0} images processed...".format(idx+1))
            img = cv2.imread(image_file_path)
            for filter_size in filter_sizes:
                blurred_image = apply_avg_blur(img, filter_size)
                cv2.imwrite(output_location + "filter_" + str(filter_size) + "/" + fn, blurred_image)
def med_blur_images(lfw_location, output_location, filter_sizes):
    print("Median Blurring all images in LFW...")
    for idx, fn in enumerate(os.listdir(lfw_location)):
        image_file_path = lfw_location + fn
        if image_file_path.endswith(".jpg"):
            if idx % 1000 == 0:
                print("{0} images processed...".format(idx+1))
            img = cv2.imread(image_file_path)
            for filter_size in filter_sizes:
                blurred_image = apply_med_blur(img, filter_size)
                cv2.imwrite(output_location + "filter_" + str(filter_size) + "/" + fn, blurred_image)
def rand_black_images(lfw_location, output_location, filter_sizes):
    print("Median Blurring all images in LFW...")
    for idx, fn in enumerate(os.listdir(lfw_location)):
        image_file_path = lfw_location + fn
        if image_file_path.endswith(".jpg"):
            if idx % 1000 == 0:
                print("{0} images processed...".format(idx+1))
            img = image_file_path
            for filter_size in filter_sizes:
                blurred_image = apply_rand_black(img, filter_size)
                cv2.imwrite(output_location + "filter_" + str(filter_size) + "/" + fn, blurred_image)


filters = [4, 6, 8]
# avg_blur_images("../data/preprocessed_lfw/cropped/", "../data/preprocessed_lfw/average_blurred/", filters)
# med_blur_images("../data/preprocessed_lfw/cropped/", "../data/preprocessed_lfw/med_blurred/", filters)
# bifilt_blur_images("../data/preprocessed_lfw/cropped/", "../data/preprocessed_lfw/bifilt_blurred/", filters)
# rand_black_images("../data/preprocessed_lfw/cropped/", "../data/preprocessed_lfw/rand_black/", filters)