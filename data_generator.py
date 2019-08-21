import numpy as np
from skimage import io 
import pandas as pd
from skimage import transform
from skimage import utils
import random
import csv

def load_label(filepath):
	return pd.read_csv(filepath)

def rotation(img):
	degree = random.uniform(-45,45)
	return transform.rotate(img, degree)

def add_noise(img):
	return utils.random_noise(img)

def flip(img):
	return img[:,::-1]

data_augmentation = {'rotate' : rotate, 'noise' : add_noise, 'flip' : flip}

def label_frequency(label):
	frequency_table = pd.crosstab(index = label[1], columns = "count")
	return frequency_table

def data_augmentation():
	print("==========Start Data Augemntation==========\n")
	filepath = "./data/train.csv"
	label = load_label(filepath)

	#get frequency_table
	frequency_table = label_frequency(label)
	
	#find imbalanced classes
	less_than_3 = frequency_table["count"] < 3 
	classes = np.asarray((table[less_than_3].index)).astype(int)
	
	for i in range(classes):
		transformations = random.sample(list(data_augmentation.keys()),2)
		
		#find path of imgs to be transformed
		class_to_transform = label[1] == classes[i]
		pathes = label[0][class_to_transform]

		path_file = "./data/train/"
		for path in pathes:
			img_path = path_file + path
			img = io.imread(img_path)
			for trans in transformations:
				transformed_img = data_augmentation[trans](img)
				new_file_path = img_path[:-4] + trans + ".png"
				io.imsave(new_file_path, transformed_img)

	print("Data Augmentation Completed !")