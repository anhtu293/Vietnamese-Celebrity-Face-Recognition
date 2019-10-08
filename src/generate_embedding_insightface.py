import argparse
import cv2
import sys
import numpy as np
sys.path.append("../backbones/insightface/deploy")
import face_model
import pandas as pd
import os
import csv

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

data_path  = '../data/train_112x112'

if __name__ == '__main__':
    model = face_model.FaceModel(args)
    img_file = []
    output_dir = './embeddings/insight/{}/{}'.format(data_path.split("/")[2], args.model.split("/")[3])
    labels = pd.read_csv("../data/train.csv")
    print(labels.head())

    for i in range(len(labels)):
        img_path = data_path + "/" + labels["image"][i]
        if not os.path.exists(img_path):
            continue
        img_origin = cv2.imread(img_path)
        print("{}/{} : {}".format(i, len(labels), img_path))

        img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
        img = np.transpose(img_origin, (2,0,1))
        embeddings = model.get_feature(img)
        np.save(output_dir + '/%s.npy'%labels["image"][i][:-4], embeddings)
        img_file.append([output_dir + '/%s.npy'%labels["image"][i][:-4], labels['label'][i]])

        img_flip = cv2.flip(img_origin, 1)
        img_flip = np.transpose(img_flip, (2,0,1))
        embeddings = model.get_feature(img_flip)
        np.save(output_dir + '/%s_flip.npy'%labels["image"][i][:-4], embeddings)
        img_file.append([output_dir + "/" + labels["image"][i][:-4] + "_flip.npy", labels['label'][i]])
    
    with open("./embeddings/insight/embs_class_{}_{}.csv".format(data_path[8:], args.model.split("/")[3]), 'a') as file:
        writer = csv.writer(file)
        writer.writerows(img_file)