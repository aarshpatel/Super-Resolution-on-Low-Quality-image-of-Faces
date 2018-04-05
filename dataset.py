import torch.utils.data as data
import torch
import os
from PIL import Image
import numpy as np
from scripts.preprocess_lfw_images import apply_gaussian_blur, pixelate


class ObfuscatedDatasetLoader(data.Dataset):

	def __init__(self, dataset_location, method, size, data_type, train_mean=None, total_num_images=None):
		""" 
		Method: type of obfuscation method - blurred or pixelated
		Size: the amount of obfuscation applied to the images
		"""
		super(ObfuscatedDatasetLoader, self).__init__()

		# get the training and test labels 
		# training - obfuscated images (either blurred or pixelated images)
		# testing - orginal grayscale image
		def get_dataset():
			""" Get all of the obfuscated images for a particular method and size """

			x_train = []
			y_train = []

			for fn in sorted(os.listdir(dataset_location + data_type + "/")):
				image_file_path = os.path.join(dataset_location + data_type + "/" + fn)
				if image_file_path.endswith(".jpg"):
					img = Image.open(image_file_path)

					y_train_img = np.array(img)
					y_train_img = y_train_img.astype(float)
					# y_train_img /= 255.0
					y_train_img = np.expand_dims(y_train_img, axis=0)
					y_train.append(y_train_img)

					# apply image obfuscation
					if method == "pixelated": img = pixelate(img, size)
					else: img = apply_gaussian_blur(img, size)

					obfuscated_img_array = np.array(img)
					obfuscated_img_array = obfuscated_img_array.astype(float)
					# obfuscated_img_array /= 255.0
					obfuscated_img_array = np.expand_dims(obfuscated_img_array, axis=0)
					x_train.append(obfuscated_img_array)

			if total_num_images is not None:
				x_train = x_train[:total_num_images]
				y_train = y_train[:total_num_images]

			x_train = np.array(x_train)
			y_train = np.array(y_train)

			return x_train, y_train


		# Do we subtract the mean image just from the training set  or we substract from the labels as well ???
		self.X_train, self.Y_train  = get_dataset()

		# self.X_train = self.X_train.astype(float)
		# self.Y_train = self.Y_train.astype(float)

		# # mean subtraction and divide by 255
		# if train_mean is None:	
		# 	# we need to calculate the train mean across the training data
		# 	self.train_mean = np.mean(self.X_train, axis=0)

		# 	self.X_train -= self.train_mean
		# 	self.Y_train -= self.train_mean

		# 	self.X_train /= 255
		# 	self.Y_train /= 255

		# else:
		# 	# substract the training mean from the val dataset and test dataset
		# 	self.X_train -= train_mean
		# 	self.X_train /= 255

		# 	self.Y_train -= train_mean
		# 	self.Y_train /= 255

		# # QUESTION: how to get back to original image, just add the train mean back to the image and scale by 255 ????

		# print(self.X_train)
		# print(self.X_train.shape)
		# print(self.Y_train.shape)

	def __getitem__(self, index):
		return self.X_train[index], self.Y_train[index]

	def __len__(self):
		return self.X_train.shape[0]
