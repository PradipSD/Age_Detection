# Age_Detection
This project combines cutting-edge deep learning techniques, such as CNNs and pre-trained models, with a user-friendly GUI implemented using Flask. By bringing together these components, it provides a convenient and accurate tool for age detection based on facial images.


CNN Model:
This code uses a modified UTKFace dataset stored in a CSV file(https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv). It preprocesses the data by dropping unnecessary columns, creating age groups, and visualizing their distribution. The image data is processed by converting pixel values and resizing images. A CNN model is defined and trained on the dataset, with model weights saved. The code sets up a Flask web application to serve the trained model, allowing users to upload images and predict the age group.

Epochs and the Optimizer play a crucial role in training a neural network effectively. The number of epochs determines the number of iterations over the dataset, while the optimizer updates the model's weights to minimize the loss function. Tuning these parameters is essential to achieve optimal performance and prevent overfitting or underfitting of the model.
In this code, the Adam optimizer is used, which is a popular and widely used optimization algorithm. The Adam optimizer adapts the learning rate for each weight based on the estimates of the first and second moments of the gradients. It combines the benefits of both the AdaGrad and RMSprop optimizers. The learning rate for the Adam optimizer is set to 0.001. The choice of optimizer can have a significant impact on the training process and the model's performance.

Pretrained Model:
This code implements an age and gender detection model using pre-trained Caffe models. It detects faces in an image using the Haar cascade classifier and then predicts the age and gender of each detected face. The code utilizes pre-trained models for age and gender detection, which have been trained on large datasets. It preprocesses the face images by resizing and normalizing them before passing them through the models. The predictions are obtained by finding the maximum probabilities from the model outputs and mapping them to corresponding labels. The code also includes a Flask web application that allows users to upload an image and obtain the predicted age group using the pre-trained models.

In the code, several files are used for the age and gender detection model. Here is an explanation of each file along with resources to download them:

1. deploy_age.prototxt and age_net.caffemodel: These files define the architecture and contain the weights of the pre-trained age detection model.

2. deploy_gender.prototxt and gender_net.caffemodel: These files define the architecture and contain the weights of the pre-trained gender detection model.
 
3. haarcascade_frontalface_alt.xml: This XML file contains the trained Haar cascade classifier for face detection. It is used to detect faces in the input image. 

#### You can download these files from this github repository: https://github.com/raunaqness/Gender-and-Age-Detection-OpenCV-Caffe/tree/master/data ###

Make sure to download these files and update the respective file paths in the code to ensure that the model can load and utilize them correctly.




Output GUI before predictoin:

![gui_before_prediction](https://github.com/PradipSD/Age_Detection/assets/100369014/1793e76c-f20e-40ab-be58-f53218a7fb34)


Output GUI after prediction:

![after_prediction](https://github.com/PradipSD/Age_Detection/assets/100369014/84d1be59-aaae-4f3f-aef9-b11660828d50)


![after_prediction_pretrained](https://github.com/PradipSD/Age_Detection/assets/100369014/3c4b9c12-5d71-4826-b385-30ca01808ca4)




