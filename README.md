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
For face detection, I've tried MTCNN but this model didn't work really well on this dataset. So I used mxnet and I got very remarkable results : only 2 faces in training set can not be detected and 60 in test set. 

I used facenet model (https://github.com/davidsandberg/facenet) with 2 pretrained models and insightface (https://github.com/deepinsight/insightface) with 2 pretrained models to create embedding vectors. 

Finally, I build a neural network with 2 hidden layers to train on embedding vectors. I had 4 models trained from 4 set of embedding vectors (2 for facenet and 2 for insightface). I used 4 models to classify test set, then I did pseudo labeling with equal weights for 4 models to get final predicts. 


## 4. Procedure :

### 1. Face detection and face alignment:
    """
        python3 ./src/face_alignment_test.py 
        python3 ./src/face_alignment_train.py
    """

### 2. Generate embedding vectors : 
    """
        python3 ./src/generate_embedding_facenet.py --model ../models/facenet/20180402-114759 
        python3 ./src/generate_embedding_facenet.py --model ../models/facenet/20180408-102900
        python3 generate_embedding_insightface.py --model ../models/insightface/model-r50-am-lfw/model,0
        python3 generate_embedding_insightface.py --model ../models/insightface/model-r100-ii/model,0
    """

### 3. Train :
    """
        python3 ./train_classifier_facenet.py --model 20180402-114759 --embeddings ./embeddings/facenet/embs_class_train_160x160_20180402-114759.csv
        python3 ./train_classifier_facenet.py --model 20180408-102900 --embeddings ./embeddings/facenet/embs_class_train_160x160_20180408-102900.csv
        python3 ./train_classifier_insightface.py --model model-r50-am-lfw --embeddings ./embeddings/insight/embs_class_train_112x112_model-r50-am-lfw.csv
        python3 ./train_classifier_insightface.py --model model-r100-ii --embeddings ./embeddings/insight/embs_class_train_112x112_model-r100-ii.csv
    """

### 4. Predict : 
    """
        python3 generate_prediction.py --model ./models/insight/model_model-r100-ii_112.h5 --embeddings ./embeddings/insight/embs_class_test_112x112_model-r100-ii.csv
        python3 generate_prediction.py --model ./models/insight/model_model-r50-am-lfw_112.h5 --embeddings ./embeddings/insight/embs_class_test_112x112_model-r50-am-lfw.csv
        python3 generate_prediction.py --model ./models/facenet/model_20180408-102900_160.h5 --embeddings ./embeddings/facenet/embs_class_test_160x160_20180408-102900.csv
        python3 generate_prediction.py --model ./models/facenet/model_20180402-102900_160.h5 --embeddings ./embeddings/facenet/embs_class_test_160x160_20180402-102900.csv
    """

### 5. Pseudo labeling:
    """
        python3 predict.py
    """

## 5. References : 
Facenet : https://github.com/davidsandberg/facenet

Insightface : https://arxiv.org/abs/1801.07698

MTCNN : https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf

1st place of the competition : https://bitbucket.org/dungnb1333/dnb-facerecognition-aivivn/src/master/

3rd place of the competition : https://github.com/vipcualo/AIVIVN2
