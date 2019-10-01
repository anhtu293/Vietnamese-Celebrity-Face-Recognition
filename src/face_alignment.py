import numpy as np
from skimage import io
import cv2
from skimage.transform import resize
from skimage import transform as trans
import pandas as pd
import time
from matplotlib import pyplot as plt
import sys
from face_detector import SSHDetector
import os
sys.path.append("../backbones/mtcnn")
from mtcnn_detector import MtcnnDetector


def face_alignment(img, size, bbox = None, landmark = None, margin = 15):
	if landmark is not None:
		if size == 112:
			src = np.array([
				[38.2946, 51.6963],
				[73.5318, 51.5014],
				[56.0252, 71.7366],
				[41.5493, 92.3655],
				[70.7299, 92.2041]], dtype = np.float32)
		elif size == 160:
			src = np.array([
				[54.706573, 73.85186],
				[105.045425, 73.573425],
				[80.036, 102.48086],
				[59.356144, 131.95071],
				[101.04271, 131.72014]], dtype = np.float32)
		dst = np.asarray(landmark).astype(np.float32)
		tform = trans.SimilarityTransform()
		tform.estimate(dst, src)
		M = tform.params[0:2,:]

	if M is None:
		if bbox is None:  # use center crop
			det = np.zeros(4, dtype=np.int32)
			det[0] = int(img.shape[1] * 0.0625)
			det[1] = int(img.shape[0] * 0.0625)
			det[2] = img.shape[1] - det[0]
			det[3] = img.shape[0] - det[1]
		else:
			det = bbox
		bb = np.zeros(4, dtype=np.int32)
		bb[0] = np.maximum(det[0] - margin / 2, 0)
		bb[1] = np.maximum(det[1] - margin / 2, 0)
		bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
		bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
		ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
		ret = cv2.resize(ret, (size, size))
		return ret
	else:
		warped = cv2.warpAffine(img, M, (size,size), borderValue = 0)
		return warped

def get_scales(img):
    TEST_SCALES = [100, 200, 300, 400]
    target_size = 400
    max_size = 1200
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [float(scale) / target_size * im_scale for scale in TEST_SCALES]
    return scales

if __name__ == '__main__':
	#delete file in train repo
	listfile = os.listdir('../data/train_112x112')
	if len(listfile) > 0:
		os.system('rm ../data/train_112x112/*')
	listfile = os.listdir('../data/train_160x160')
	if len(listfile) > 0:
		os.system('rm ../data/train_160x160/*')
	listfile = os.listdir('../data/train_160x160')
	if len(listfile) > 0:
		os.system('rm ../data/train_unknown/*')

	print("Start Face Alignment\n")
	labels = pd.read_csv('../data/train.csv')
	print(labels.head(5))
	print(labels.shape)
	face_detector = SSHDetector(prefix = "../models/ssh/sshb", ctx_id= -1, epoch = 0, test_mode = True)
	landmark_detector = MtcnnDetector(model_folder = "../backbones/mtcnn/model")
		
	for i in range(labels.shape[0]):
		print("Image : {} ----- File : {} ------\n".format(i+1, labels['image'][i]))
		img = io.imread('../data/train/' + labels['image'][i])
		scales = get_scales(img)
		results = face_detector.detect(img, scales = scales, threshold = 0.2)
		if len(results) != 0:
			landmarks = landmark_detector.get_landmark(img, results)
			if landmarks != None:
				_, points = landmarks
				points_array = points[0,:].reshape((2,5)).T
				new_face_112x112 = face_alignment(img, 112, landmark = points_array, bbox = results[0])
				new_face_160x160 = face_alignment(img, 160, landmark = points_array, bbox = results[0])
				
				io.imsave('../data/train_112x112/'+labels['image'][i], new_face_112x112)
				io.imsave('../data/train_160x160/'+labels['image'][i], new_face_160x160)
			else:
				io.imsave('../data/train_unknown/'+labels['image'][i], img)
		else:
			io.imsave('../data/train_unknown/'+labels['image'][i], img)

	print("Face alignment completed !")


