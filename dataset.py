import torch.utils.data as data
import torch
import os
from PIL import Image
import numpy as np
from scripts.preprocess_lfw_images import apply_gaussian_blur, pixelate


class ObfuscatedDatasetLoader(data.Dataset):
	def __init__(self, dataset_location, method, size, grayscale, data_type, transform=None):
		""" 
		Method: type of obfuscation method - blurred or pixelated
		Size: the amount of obfuscation applied to the images
		"""
		super(ObfuscatedDatasetLoader, self).__init__()

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
					if grayscale:
						y_train_img = np.expand_dims(y_train_img, axis=0)
					y_train.append(y_train_img)

					# apply image obfuscation
					if method == "pixelated": img = pixelate(img, size)
					else: img = apply_gaussian_blur(img, size)

					obfuscated_img_array = np.array(img)
					obfuscated_img_array = obfuscated_img_array.astype(float)
					if grayscale:
						obfuscated_img_array = np.expand_dims(obfuscated_img_array, axis=0)
					x_train.append(obfuscated_img_array)

			x_train = np.array(x_train)
			y_train = np.array(y_train)

			print "Dataset Type: ", data_type
			print x_train.shape
			print y_train.shape

			return x_train, y_train


		self.X_train, self.Y_train  = get_dataset()
		self.transform = transform

	def __getitem__(self, index):
		x = self.X_train[index]
		y = self.Y_train[index]

		if self.transform:
			x = self.transform(x)
			y = self.transform(y)
		return x, y  

	def __len__(self):
		return self.X_train.shape[0]
