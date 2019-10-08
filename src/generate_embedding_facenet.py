import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append("../backbones/facenet/src")
import facenet
import argparse
import csv
import cv2
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(description = 'facenet')
parser.add_argument("--data", default = '../data/train_160x160', help = 'path to dataset')
parser.add_argument("--model", default = '../models/facenet/20180402-114759', help = 'path to load pretrained model')
parser.add_argument('--image_size', type = int, help = 'Image size (h,w) to input model', default = 160)
args = parser.parse_args()

print(args.model)

def standarize_img(x):
	mean = np.mean(x)
	std = np.std(x)
	std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
	y = np.multiply(np.subtract(x, mean), 1/std_adj)
	return y

if __name__ == '__main__':
	with tf.Graph().as_default():
		with tf.Session() as sess:
			#load model
			facenet.load_model(args.model)
			
			img_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			
			img_file = []

			output_dir = './embeddings/facenet/{}/{}'.format(args.data.split("/")[2], args.model.split("/")[3])
			labels = pd.read_csv("../data/train.csv")
			print(labels.head())

			for i in range(len(labels)):
				img_path = args.data + "/" + labels['image'][i]
				if not os.path.exists(img_path):
					continue
				img_origin = cv2.imread(img_path)
				
				print("{}/{} : {}".format(i, len(labels), img_path))
				img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
				img = standarize_img(img_origin)
				img = np.expand_dims(img, axis = 0)
				feed_dict = {img_placeholder : img, phase_train_placeholder : False}
				embed = sess.run(embeddings, feed_dict = feed_dict)
				np.save(output_dir + '/%s.npy'%labels["image"][i][:-4], embed)
				img_file.append([output_dir + '/%s.npy'%labels["image"][i][:-4], labels['label'][i]])

				img_flip = cv2.flip(img_origin, 1)
				img_flip = standarize_img(img_flip)
				img_flip = np.expand_dims(img_flip, axis = 0)
				feed_dict = {img_placeholder : img_flip, phase_train_placeholder : False}
				embed = sess.run(embeddings, feed_dict = feed_dict)

				np.save(output_dir + '/%s_flip.npy'%labels["image"][i][:-4], embed)
				img_file.append([output_dir + "/" + labels["image"][i][:-4] + "_flip.npy", labels['label'][i]])

			with open("./embeddings/facenet/embs_class_{}_{}.csv".format(args.data.split("/")[2], args.model.split("/")[3]), 'a') as file:
				writer = csv.writer(file)
				writer.writerows(img_file)