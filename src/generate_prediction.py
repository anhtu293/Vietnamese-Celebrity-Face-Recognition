import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description = "arguments for output model prediction")
# ./models/facenet/....
parser.add_argument("--model", default = "", help = "path to model")
parser.add_argument("--embeddings", default = "", help = "path to csv index to embedding files")
args = parser.parse_args()

if __name__ == "__main__":
    model = load_model(args.model)
    labels = pd.read_csv(args.embeddings)
    labels.columns = ["image"]
    print(labels.head)
    probs = []
    cols = ["image"]
    for i in range(1, 1001):
        cols.append("class_{}".format(i))
    for i in range(labels.shape[0]):
        print(i)
        if not os.path.exists(labels["image"][i]):
            p = [0 for i in range(1,1000)]
            probs.append([labels["image"][i], p]) 
        embs = np.load(labels["image"][i]).reshape(-1, 512)
        pred = model.predict(embs, verbose = 1)   
        p = []
        p.append(labels["image"][i])     
        for j in range(1000):
            p.append(pred[0][j])
        probs.append(p)
    probs = pd.DataFrame(probs)
    #print(probs.head)
    probs.columns = [cols]
    print(probs.head)
    probs.to_csv("./results/" + args.model.split("/")[2] + "_" + args.model.split("/")[3] + ".csv")


