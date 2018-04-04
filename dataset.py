import torch.utils.data as data
import torch
import os
from PIL import Image
import numpy as np


class ObfuscatedDatasetLoader(data.Dataset):

	def __init__(self, gt_images_location, method, size):
		""" 
		Method: type of obfuscation method - blurred or pixelated
		Size: the amount of obfuscation applied to the images
		"""
		super(ObfuscatedDatasetLoader, self).__init__()
		
		# get the training and test labels 
		# training - obfuscated images (either blurred or pixelated images)
		# testing - orginal grayscale image

		def get_all_obfuscated_dataset(method, size):
			""" Get all of the obfuscated images for a particular method and size """
			obfuscation_method_dataset = []

			for fn in os.listdir("./data/lfw_preprocessed/" + method + "/size_" + str(size) + "/"):
				image_file_path = os.path.join("./data/lfw_preprocessed/" + method + "/size_" + str(size) + "/", fn)
				if image_file_path.endswith(".jpg"):
					img = Image.open(image_file_path)
					img_array = np.array(img)
					img_array = np.expand_dims(img_array, axis=0)

				obfuscation_method_dataset.append(img_array)

			obfuscation_method_dataset = np.array(obfuscation_method_dataset)	

			return obfuscation_method_dataset

		def get_ground_truth_dataset(dataset_location):
			""" Get all of the ground truth images """ 
			ground_truth_images = []

			for fn in os.listdir(dataset_location):
				image_file_path = os.path.join(dataset_location, fn)
				if image_file_path.endswith(".jpg"):
					img = Image.open(image_file_path)
					img_array = np.array(img)
					img_array = np.expand_dims(img_array, axis=0)

				ground_truth_images.append(img_array)

			ground_truth_images = np.array(ground_truth_images)

			return ground_truth_images

		self.X_train = get_all_obfuscated_dataset(method, size)
		self.Y_train = get_ground_truth_dataset(gt_images_location)

	def __getitem__(self, index):
		return self.X_train[index], self.Y_train[index]

	def __len__(self):
		return self.X_train.shape[0]

