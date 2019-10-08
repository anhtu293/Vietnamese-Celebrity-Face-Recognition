# Vietnamese celebrities face recognition contest on www.aivivn.com

## 1. Data analysis

There is obviously imbalanced phenomenon in this data set : Some classes have only 2 or 3 examples, others have up to 10 examples. To solve this problem, I find classes which have less than 5 examples and do some random transformations : with classes which have 2 examples, 4 transformations were implemented, with classes which have 3 examples, 3 transformations were implemented and with classes which have 4 examples, 2 transformations were implemented. Therefor, all the classes basically have at least 6 example for training. 

With the limited number of data, the idea is to use a pretrained model for face recognition. After having done some research, I found a pretrained model called FaceNet that would be useful. The limitation of this model is that the data is not faces of asian people meanwhile our purpose is to recognize vietnamese celebrities. 

## 2. Data preprocessing
I divided the data set into 3 parties : trainning set, validation set and test set with the proportion : 60%, 20%, 20% to prepare for the training.

Like other face recognition projects, the procedure is : face detection, face alignment, embedding vectors generation.
In this project, I've tried 2 face detectors :  MTCNN and Mxnet. MTCNN showed very bad results : about 200 images in training set and 600 images in test set that can not be detected by MTCNN. This problem can be explained that many faces in this data were looking down or other ways, some others are in black and white. So I tried the face detector of mxnet with threshold = 0.2. The result was remarkable compared to MTCNN. 

After face detection, the faces detected by the Mxnet detector were passed MTCNN to find the landmarks. In each image, there were 5 landmark points were found and an array of 10 coordonnates were returned by MTCNN. I used these landmarks to align the faces. 

Finally, I used 2 architectures with 3 models to calculate embedding vectors : insight and facenet. To create submission, the pseudo voting was used.
## 3. Models 
Insight

Facenet

Mxnet

MTCNN
## 4. Tuning

## TODO
[x] find pretrained model OK

[x] data augmentation OK

[x] data split OK

[x] face alignment OK => Need to improve. MTCNN doenst work very well. Try MXnet SSH instead of MTCNN, fix bug in mtcnn_detector get_landmark. ref : https://stackoverflow.com/questions/54453957/not-able-to-process-some-images-for-face-detection-using-mtcnn-mehtod-implement

[ ] extract embedding vectors

[ ] pseudo training

[ ] pseudo voting

[ ] find threshold for unknown class
	