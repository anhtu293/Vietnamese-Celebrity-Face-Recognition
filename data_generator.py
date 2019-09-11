import numpy as np
from skimage import io 
import pandas as pd
from skimage import transform
from skimage import util
import random
import csv

def load_label(filepath):
	return pd.read_csv(filepath)

def rotate(img):
	degree = random.uniform(-45,45)
	return transform.rotate(img, degree)

def add_noise(img):
	return util.random_noise(img)

def flip(img):
	return img[:,::-1]

data_augmentation = {'rotate' : rotate, 'noise' : add_noise, 'flip' : flip}

def label_frequency(label):
	frequency_table = pd.crosstab(index = label["label"], columns = "count")
	return frequency_table

print("==========Start Data Augemntation==========\n")
filepath = "./data/train.csv"
label = load_label(filepath)

#get frequency_table
frequency_table = label_frequency(label)
	
#find imbalanced classes
less_than_3 = frequency_table["count"] < 5 
classes = np.asarray((frequency_table[less_than_3].index)).astype(int)
	
name_class = []

for i in range(classes.shape[0]):
	if frequency_table["count"][classes[i]] == 2:
		transformations = data_augmentation
	if frequency_table["count"][classes[i]] == 3:
		transformations = random.sample(list(data_augmentation.keys()),3)
	if frequency_table["count"][classes[i]] == 4:
		transformations = random.sample(list(data_augmentation.keys()),2)

	transformations = random.sample(list(data_augmentation.keys()),2)
	
	#find path of imgs to be transformed
	class_to_transform = label["label"] == classes[i]
	paths = label["image"][class_to_transform]

	path_file = "./data/train/"
	for path in paths:
		img_path = path_file + path
		img = io.imread(img_path)
		for trans in transformations:
			transformed_img = data_augmentation[trans](img)
			new_file_path = img_path[:-4] + "_" + trans + ".png"
			io.imsave(new_file_path, transformed_img)

			name_image = path[:-4] + "_" + trans + ".png"
			name_file = [name_image, classes[i]]
			name_class.append(name_file)

#add index to label file
with open(filepath, 'a') as file:
	writer = csv.writer(file)
	writer.writerows(name_class)

print("Data Augmentation Completed !")