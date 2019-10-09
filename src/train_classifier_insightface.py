import tensorflow as tf
import pickle
import numpy as np
import os
import pandas as pd
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

import argparse

parser = argparse.ArgumentParser(description = "train classifier of insight model")

parser.add_argument("--model", default = "20180402-114759", help = "model name")
parser.add_argument("--embeddings", default = "./embeddings/insight/embs_class_train_112x112_model-r50-am-lfw.csv", help = "path to file index embeddings for train")
parser.add_argument("--size", default = 112, help = 'image size')
args = parser.parse_args()

INPUT_SHAPE = 512
LEARNING_RATE = 0.001
BATCH_SIZE = 64

def build_model():
    model = Sequential()
    model.add(Dense(2048, input_dim = INPUT_SHAPE, activation = "relu"))
    model.add(Dropout(0.25))
    model.add(Dense(1000, activation = "softmax"))
    return model

def DataGenerator(labels, classes, batch_size):
    i = 0
    while True:
        embeddings_batch = np.array([],dtype = np.float32).reshape(0,512)
        class_batch = []
        for b in range(batch_size):
            if i == labels.shape[0]:
                i = 0
            sample = np.load(labels["emb"][i]).reshape(-1,512)
            cl = classes[i]
            embeddings_batch = np.vstack((embeddings_batch, sample))
            class_batch.append(cl)
            i += 1
        yield np.array(embeddings_batch), np.array(class_batch)

def convert2categorical(labels):
    categories = []
    for index, row in labels.iterrows():
        categories.append(to_categorical(row[1], num_classes = 1000))
    return np.array(categories)

if __name__ == '__main__':
    model = build_model()
    labels = pd.read_csv(args.embeddings)
    labels.columns = ['emb', 'class']
    print(labels.head)
    categories = convert2categorical(labels)
    """
    labels_train = labels.sample(frac = 0.8, random_state = 0)
    labels_val = labels.drop(labels_train.index)
    categories_train = convert2categorical(labels_train)
    categories_val = convert2categorical(labels_val)
    
    labels_train.reset_index(inplace = True)
    labels_val.reset_index(inplace = True)
    print(labels_train.head)
    print(labels_val.head)
    print(categories_train.shape)
    print(categories_val.shape)
    """

    train_generator = DataGenerator(labels, categories, BATCH_SIZE)
    #val_generator = DataGenerator(labels_val, categories_val, BATCH_SIZE)

    optimizer = Adam(LEARNING_RATE)
    metric = tf.keras.metrics.CategoricalAccuracy()
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = [metric])

    callbacks = [
		tf.keras.callbacks.EarlyStopping(patience = 8, monitor = 'val_acc', restore_best_weights = True),
		tf.keras.callbacks.ModelCheckpoint(filepath ="./checkpoints/insight/"+args.model+"/weights-epoch{epoch:02d}-loss{val_loss:.2f}.h5")
		]
    history = model.fit_generator(generator = train_generator, epochs = 100, verbose = 1, steps_per_epoch = len(labels)/BATCH_SIZE)
    
    model.save("./models/insight/model_" + args.model + "_" + str(args.size) + ".h5")