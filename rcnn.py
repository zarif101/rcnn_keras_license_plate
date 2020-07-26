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
from helper_functions import load_data,non_max_suppression_fast
import models
#I USE PLAIDML BECAUSE I RUN AN AMDGPU AND I CANNOT USE TENSORFLOW
#UNLESS YOU SPECIFICALLY KNOW THAT YOU RUN PLAIDML INSTEAD OF TENSORFLOW...
#DELETE THESE IMPORTS AND SIMPLY IMPORT TENSORFLOW/KERAS AS YOU NORMALLY WOULD
import plaidml.keras
plaidml.keras.install_backend()
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
from keras.optimizers import Adam
from keras.models import load_model
#----

def rcnn(image,base_model_name):
    model = load_model(base_model_name)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    results = ss.process()
    copy = image.copy()
    copy2 = image.copy()
    positive_boxes = []
    probs = []

    for box in results:
        x1 = box[0]
        y1 = box[1]
        x2 = box[0]+box[2]
        y2 = box[1]+box[3]

        roi = image.copy()[y1:y2,x1:x2]
        roi = cv2.resize(roi,(128,128))
        roi_use = roi.reshape((1,128,128,3))
        class_pred = model.predict_classes(roi_use)[0][0]

        if class_pred == 1:
            prob = model.predict(roi_use)[0][0]
            if prob > 0.98:
                positive_boxes.append([x1,y1,x2,y2])
                probs.append(prob)
                cv2.rectangle(copy2,(x1,y1),(x2,y2),(255,0,0),5)

    cleaned_boxes = non_max_suppression_fast(np.array(positive_boxes),0.1,probs)
    total_boxes = 0
    for clean_box in cleaned_boxes:
        clean_x1 = clean_box[0]
        clean_y1 = clean_box[1]
        clean_x2 = clean_box[2]
        clean_y2 = clean_box[3]
        total_boxes+=1
        cv2.rectangle(copy,(clean_x1,clean_y1),(clean_x2,clean_y2),(0,255,0),3)
    #plt.imshow(copy2)
    plt.imshow(copy)
    plt.show()

def main(image_name,model_name):
    test_img = cv2.imread(image_name)
    rcnn(image=test_img,base_model_name=model_name)

main('many_plates.jpg','base_model.h5')
