import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import xml.etree.ElementTree as ET
import numpy as np
import PIL.Image as Image
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from helper_functions import load_data
import models
#I USE PLAIDML BECAUSE I RUN AN AMDGPU AND I CANNOT USE TENSORFLOW
#UNLESS YOU SPECIFICALLY KNOW THAT YOU RUN PLAIDML INSTEAD OF TENSORFLOW...
#DELETE THESE IMPORTS AND SIMPLY IMPORT TENSORFLOW/KERAS AS YOU NORMALLY WOULD
import plaidml.keras
plaidml.keras.install_backend()
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
from keras.optimizers import Adam
#----

def main():
    X,y = load_data()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3) #use fit so will use validation_split

    model = models.get_model_1(input_shape=(128,128,3))
    optimizer=Adam(lr=.0005)
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

    history = model.fit(X_train,y_train,validation_split=0.15,epochs=8)

    print(model.evaluate(X_test,y_test))
    model.save('base_model.h5')


main()
