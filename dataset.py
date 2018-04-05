import torch.utils.data as data
import torch
import os
from PIL import Image
import numpy as np
from scripts.preprocess_lfw_images import apply_gaussian_blur, pixelate


class ObfuscatedDatasetLoader(data.Dataset):

	def __init__(self, dataset_location, method, size, data_type):
		""" 
		Method: type of obfuscation method - blurred or pixelated
		Size: the amount of obfuscation applied to the images
		"""
		super(ObfuscatedDatasetLoader, self).__init__()

		if method == "pixelated":
			print("Apply pixelation with size = ", size)
		else:
			print("Apply gaussian blur with radius = ", size)
		
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
					y_train.append(np.array(img))

					# apply image obfuscation
					if method == "pixelated":
						img = pixelate(img, size)
					else:
						img = apply_gaussian_blur(img, size)

					obfuscated_img_array = np.array(img)
					obfuscated_img_array = np.expand_dims(obfuscated_img_array, axis=0)

				x_train.append(obfuscated_img_array)


			x_train = np.array(x_train)
			y_train = np.array(y_train)

			return x_train, y_train


		self.X_train, self.Y_train  = get_dataset()

	def __getitem__(self, index):
		return self.X_train[index], self.Y_train[index]

	def __len__(self):
		return self.X_train.shape[0]

