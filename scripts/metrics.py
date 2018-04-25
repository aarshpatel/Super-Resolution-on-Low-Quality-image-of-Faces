""" Implementation of different metrics """ 

import numpy as np
import math
from skimage.measure import compare_psnr, compare_ssim
from math import log10

def calc_ssim(img1, img2):
    """ Compute the structural similarity between two images (SSIM) """
    return compare_ssim(img1, img2, multichannel=True)

def calc_psnr(mse,size):
	"""Calculate the psnr (Peak Signal Noise Ratio)"""
	return (20.0 * log10(255.0)) - (10.0 * log10(mse/size))