import pandas as pd
import os
import numpy as np
from sklearn.model_selection import KFold

NUMBER_OF_FOLDS = 3

if __name__ == '__main__':
	labels = pd.read_csv("./data/train.csv")

	#train set
	train_df = labels.sample(frac = 0.6, random_state = 0)

	#validation set, test set 
	validation_test = labels.drop(train_df.index)
	validation_df = validation_test.sample(frac = 0.5, random_state = 1)
	test_df = validation_test.drop(validation_df.index)

	#remove old index
	train_df = train_df.reset_index(drop = True)
	validation_df = validation_df.reset_index(drop = True)
	test_df = test_df.reset_index(drop = True)

	#save 
	train_df.to_csv("./data/train_df.csv", index = False)
	validation_df.to_csv("./data/validation_df.csv", index = False)
	test_df.to_csv("./data/test_df.csv", index = False)

	print(train_df.head(10))
	print(validation_df.head(10))
	print(test_df.head(10))