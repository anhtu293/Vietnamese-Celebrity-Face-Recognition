# Vietnamese celebrities face recognition contest on www.aivivn.com

## 1. Data analysis

There is obviously imbalanced phenomenon in this data set : Some classes have only 2 or 3 examples, others have up to 10 examples. To solve this problem, I find classes which have less than 5 examples and do some random transformations : with classes which have 2 examples, 4 transformations were implemented, with classes which have 3 examples, 3 transformations were implemented and with classes which have 4 examples, 2 transformations were implemented. Therefor, all the classes basically have at least 6 example for training. 

With the limited number of data, the idea is to use a pretrained model for face recognition. After having done some research, I found a pretrained model called FaceNet that would be useful. The limitation of this model is that the data is not faces of asian people meanwhile our purpose is to recognize vietnamese celebrities. 

## 2. Data preprocessing

## 3. Models 

## 4. Tuning

## TODO
	- find pretrained model OK
	- data augmentation OK
	- data split OK
	- face alignment OK => Need to improve. MTCNN doenst work very well 
	- extract embedding vectors
	- pseudo training
	- pseudo voting
	- find threshold
	- unknown class
	