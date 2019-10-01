import tensorflow as tf
import numpy as np
import os
import sys
import facenet
import align.detect_face
import argparse

parser = argparse.ArgumentParser(description = 'facenet')
parser.add_argument("--model", default = '', help = 'path to load pretrained model')
parser.add_argument('--image_size', type = int, help = 'Image size (h,w) to input model', default = 160)
args = parser.parse_args()

print(args.model)

def prewhiten(x):
	mean = np.mean(x)
	std = np.std(x)
	std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
	y = np.multiply(np.subtract(x, mean), 1/std_adj)
	return y

if __name__ = '__main__':
	with tf.Graph().as_default():
		with tf.Session() as sess:
			#load model
			facenet.load_model(args.model)
			
