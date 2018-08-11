#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 18:12:13 2018

@author: francesctarres
"""

# Tensorflow and tf.keras libraries
import tensorflow as tf
from tensorflow import keras

# Other helper libraries

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Reading data from MNIST libraries

mnist_data = keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels) = mnist_data.load_data()

# We also normalize the images

train_images = train_images / 255.0 
test_images = test_images / 255.0 

# We can verify the size of the training samples (60000) 
#and the size of the images (28 x 28)

print(train_images.shape)

# Once images have been load we can vverify one of the training examples 
# and its class

plt.figure()
plt.imshow(train_images[100],cmap='gray')

print(train_labels[100])


# Representing some of the training examples with their labels

class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 
               'Five', 'Six', 'Seven', 'Eight', 'Nine']

# Representing and array of random samples of images and labels
# in the MNIST training database

rand_sampling = np.random.random_integers(0,60000,25)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[rand_sampling[i]], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[rand_sampling[i]]])
    
plt.savefig('ejemplo_trining.png')

# Defining & building a model of linear regression (one layer of 10 neurons)

model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(10,activation=tf.nn.softmax)
])
    
# Compiling the model

model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model

model.fit(train_images, train_labels, epochs=5)    
    