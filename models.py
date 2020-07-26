#I USE PLAIDML BECAUSE I RUN AN AMDGPU AND I CANNOT USE TENSORFLOW
#UNLESS YOU SPECIFICALLY KNOW THAT YOU RUN PLAIDML INSTEAD OF TENSORFLOW...
#DELETE THESE IMPORTS AND SIMPLY IMPORT TENSORFLOW/KERAS AS YOU NORMALLY WOULD
import plaidml.keras
plaidml.keras.install_backend()
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
from keras.optimizers import Adam
#----

def get_model_1(input_shape):
    model = Sequential()

    model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=input_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))


    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(rate=0.35))
    model.add(Dense(1,activation='sigmoid'))

    return model
