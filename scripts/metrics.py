""" Implementation of different metrics """ 

import numpy as np
import math
from skimage.measure import compare_psnr, compare_ssim

def psnr(img1, img2):
    """ Compute the Peak Signal Noise Ratio between two images (PSNR) """
    return compare_psnr(img1, img2)

def ssim(img1, img2):
    """ Compute the structural similarity between two images (SSIM) """
    return compare_ssim(img1, img2, multichannel=True)

