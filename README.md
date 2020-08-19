# Project for subject MLE

Select a dataset from Kaggle and write a machine learning application on Colab:

- Using best-practices for building deep learning applications: Hyperparameter tuning,
Regularization and Optimization.

- Showing how to diagnose errors in a machine learning system, and be able to prioritize the most
promising directions for reducing error.

# About American Sign Language (ASL)
This project use DNN network with Tensorflow v1.14 framework to train. 
The topic here is about American Sign language which is the primary language used by many deaf individuals in North America. 

In this notebook, I will train a network to classify images of American Sign Language (ASL) letters. After loading, examining, and preprocessing the data, I will train the network and test its performance.

## 1. Dataset
The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING. The folder which contains those images is datasets/asl_alphabet_train.

Because my computer is not qualified for long-time-training enough so I use a tool to get randomly 5689 images from the dataset (approximately 200 images / 1 label) to use, which is in the datasets/asl_alphabet_dev.

(Kaggle: https://www.kaggle.com/grassknoted/asl-alphabet)

At first I chose to load the images from the dataset to X_train_origin, X_test_origin, Y_train_origin, Y_test_origin using utils function I wrote separately. 

In the loading phase, I change the images's size from 200x200 to 64x64 and use train_test_split function which from sklearn.model_selection to split these images to training set and test set with the accuracy 70:30. After that, I saved these vector images and its labels to X_train_origin, X_test_origin, Y_train_origin, Y_test_origin.

## 2. Visualize the training data

I sketch the figure 3 x 12 images with its label which randomly got from the dataset.

## 3. Handle the images in the dataset

In this phase, the training and test images will be flatten and converted to one hot matrices. The convert_to_one_hot I wrote separately in utils file.

## 4. Training model

I chose DNN network using Tensorflow ver 1.14 to train. The model is **LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX**. The **SIGMOID** output layer has been converted to a SOFTMAX. A SOFTMAX layer generalizes SIGMOID to when there are more than two classes. The process is:

- Create placeholders for X and Y.
- Initializes parameters to build a neural network with tensorflow.
- Forward propagation in tensorflow.
- Compute cost

The last model implements a three-layer tensorflow neural network: **LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX** will use the above functions with these hyperparameters below:

- learning_rate (alpha) = 0.0001,
- num_epochs = 1500
- minibatch_size = 32
- lambd = 0.001 (regularization parameter)

To avoid bias, I think approximate 4k images is enough. Also, in this model I use Adam optimizer.

### 4.1 - Create placeholders
Create placeholders for `X` and `Y`.

### 4.2 - Initializing the parameters
Initializes parameters to build a neural network with tensorflow.

### 4.3 - Forward propagation in tensorflow 
The function will take in a dictionary of parameters and it will complete the forward pass.

### 4.4 Compute cost

### 4.5 - Building the model
Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

Train Accuracy: 1.0
Test Accuracy: 0.92852956
The train accuracy and test accuracy is different from each other ~ 7% so I decided to add L2 regularization to reduce overfitting.

After regularization:

Train Accuracy: 1.0
Test Accuracy: 0.92852956

Since regularization can not help reducing overfitting, I think this result can be accepted.
