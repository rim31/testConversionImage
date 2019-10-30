# -*- coding: utf-8 -*-
"""ConversionImage.ipynb
https://colab.research.google.com/

if u don't want to set the environment in your computer

"""

import pandas as pd
import tensorflow.keras as K
import matplotlib.pyplot as pl
from matplotlib.pyplot import imshow

#ls
# import images in the right folder

from keras.applications.resnet50 import ResNet50
from keras.models import Model

model = ResNet50(include_top=False)
input = model.layers[0].input

# Remove the average pooling layer
output = model.layers[-2].output
headless_conv = Model(inputs=input, outputs=output)

import cv2 as cv

img = cv.imread('10.jpg')

headless_conv.predict(img)

# function to convert image to a model RELU sequential (cv)
def create_model_relu():
    model = Sequential() 

    model.add(Conv2D(16, (3, 3), input_shape=(224,224,3), padding="same", activation='relu'))
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    return model

img.shape

from tensorflow.keras.models import * 
import tensorflow
from tensorflow.keras.layers import *

import numpy as np

#1.   Élément de liste
#2.   Élément de liste


np.expand_dims(img, axis=0).shape

import PIL
from PIL import Image
from glob import glob

imgs = glob("*.jpg")


imgLst = [cv.imread(x) for x in imgs]

imgLst = [cv.resize(image, (224, 224)) for image in imgLst]

model = create_model_relu()

imgLst = np.array(imgLst)
imgLst.shape

model.predict(imgLst)

model.predict(imgLst).shape

model.predict(imgLst)

