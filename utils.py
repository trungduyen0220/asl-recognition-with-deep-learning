import random
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

train_directory = "datasets/asl_alphabet_train_small"
test_directory = "datasets/asl_alphabet_test"

def load_data():
	X_train, Y_train = load_image(load_directory = train_directory)
	X_test, Y_test = load_image(load_directory = test_directory)

	Y_train = Y_train.reshape((1, Y_train.shape[0]))
	Y_test = Y_test.reshape((1, Y_test.shape[0]))

	return X_train, X_test, Y_train, Y_test

def load_image(load_directory):
	images=[]
	labels=[]
	size=64,64
	print()
	print("Loading... " + load_directory + ": ",end = "")
	for folder_index, folder in enumerate(os.listdir(load_directory)):
		print(folder,end='|')
		for image in os.listdir(load_directory + "/" + folder):
			temp_img = cv2.imread(load_directory + '/' + folder +'/' + image)
			temp_img = cv2.resize(temp_img,size)
			images.append(temp_img)
			labels.append(folder_index)
	# Normalize image vectors
	images = np.array(images)/255.0
	images = images.astype('float32')
	labels = np.array(labels)
	return images, labels


def convert_to_one_hot(Y, C):
	Y = np.eye(C)[Y.reshape(-1)].T
	return Y