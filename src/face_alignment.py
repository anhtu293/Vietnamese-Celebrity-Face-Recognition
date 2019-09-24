import numpy as np
from skimage import io
from mtcnn.mtcnn import MTCNN
import cv2
from skimage.transform import resize
from skimage import transform as trans
import pandas as pd
import time
from matplotlib import pyplot as plt


def face_aligment(img, size, bbox = None, landmark = None, margin = 15):
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
		tform.estimate(dst = dst, src = src)
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
		if len(image_size) > 0:
			ret = cv2.resize(ret, (image_size[1], image_size[0]))
		return ret
	else:
		warped = cv2.warpAffine(img, M, (size,size), borderValue = 0)
		return warped

if __name__ == '__main__':
	labels = pd.read_csv('../data/train.csv')
	print(labels.head(5))
	print(labels.shape)
	detector = MTCNN()

	for i in range(labels.shape[0]):
		img = io.imread('../data/train/' + labels['image'][i])
		result = detector.detect_faces(img)

		if len(result) == 0:
			io.imsave('../data/train_unknown/'+labels['image'][i], img)
			continue

		keypoints = result[0]["keypoints"]
		landmark = np.zeros((5,2))
		keys = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']

		for j in range(landmark.shape[0]):
			landmark[j,:] = keypoints[keys[j]]
		#size = 112x112
		new_face_112x112 = face_aligment(img, 112, landmark = landmark, bbox = result[0]["box"])
		#size = 160x160
		new_face_160x160 = face_aligment(img, 160, landmark = landmark, bbox = result[0]["box"])

		io.imsave('../data/train_112x112/'+labels['image'][i], new_face_112x112)
		io.imsave('../data/train_160x160/'+labels['image'][i], new_face_160x160)
