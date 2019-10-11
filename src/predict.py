import pandas as pd
import numpy as np
import sys
import os

path1 = "./results/facenet_model_20180402-114759_160.h5.csv"
path2 = "./results/facenet_model_20180408-102900_160.h5.csv"
path3 = "./results/insight_model_model-r50-am-lfw_112.h5.csv"
path4 = "./results/insight_model_model-r100-ii_112.h5.csv"
if __name__ == '__main__':
    probs1 = pd.read_csv(path1)
    probs2 = pd.read_csv(path2)
    probs3 = pd.read_csv(path3)
    probs4 = pd.read_csv(path4)
    results = []
    for i in range(probs1.shape[0]):
        print("{}/{}".format(i, probs1.shape[0]))
        avg_probs = np.array([(probs1["class_{}".format(x)][i] + probs2["class_{}".format(x)][i] + 
            probs3["class_{}".format(x)][i] + probs4["class_{}".format(x)][i])/4 for x in range(1,1001)])
        avg_probs = np.argsort(avg_probs)[0:4]
        if avg_probs[0] == 0:
            res = "1001"
        l = []
        for i in range(4):
            l.append(str(avg_probs[i]+1))
        res = " ".join(l)
        results.append([probs1["image"][i], res])
    results = pd.DataFrame(results)
    results.to_csv("./results/final.csv")