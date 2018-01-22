# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 18:51:49 2018

@author: Borislav
"""

from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D,Flatten, Activation, Input
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau


#%%

def ConvBlock(model, n_conv, n_filters):
    
    for _ in range(n_conv):
        model.add(Conv2D(n_filters, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    return model

def ConvBlockNorm(model, n_filters, n_conv=1, input_shape=()):
    
    for i in range(n_conv):
        if len(input_shape) > 0 and i == 0:
            model.add(Conv2D(n_filters, kernel_size=(3, 3), padding='same', input_shape=input_shape, use_bias=False))
        else:
            model.add(Conv2D(n_filters, kernel_size=(3, 3), padding='same', use_bias=False))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    return model


def DenseTop(model, units, dropout):
    
    model.add(Flatten())
    for u in units:
        model.add(Dense(u, activation='relu'))
        if dropout > 0.0:
            model.add(Dropout(dropout))
    # Output 
    model.add(Dense(1, activation="sigmoid"))
    return model



def getVgg():
    """"
    Takes 1h / epoch
    
    """
    
    model = Sequential()
    
    # Input layer 
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 2)))
    
    # Blocks
    model = ConvBlock(model, 1, 64)
    model = ConvBlock(model, 2, 128)
    model = ConvBlock(model, 3, 256)
    
    # Top
    model = DenseTop(model, [512, 256], 0.3)
    
    return model




def getNormModel():
    
    model = Sequential()
    
    # Convolution + ReLU + BatchNorm + Maxpool
    model = ConvBlockNorm(model, 32, 1, input_shape=(75, 75, 2))
    model = ConvBlockNorm(model, 64, 1)
    model = ConvBlockNorm(model, 128, 1)
    model = ConvBlockNorm(model, 128, 1)
    model = ConvBlockNorm(model, 128, 1)
    
    # Top
    model = DenseTop(model, [256, 256], 0.3)
    
    return model
    

def getModel():
    #Build keras model
    
    model=Sequential()
    
    # CNN 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 2)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    #CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    # You must flatten the data for the dense layers
    model.add(Flatten())

    #Dense 1
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    #Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


#%%




model = getNormModel()
model.summary()



#%%


def ConvBlockNorm(model, n_filters, n_conv=1, dropout=0.0, input_shape=()):
    
    for i in range(n_conv):
        if len(input_shape) > 0 and i == 0:
            model.add(Conv2D(n_filters, kernel_size=(3, 3), padding='same', input_shape=input_shape, use_bias=False))
        else:
            model.add(Conv2D(n_filters, kernel_size=(3, 3), padding='same', use_bias=False))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    if dropout > 0.0:
        model.add(Dropout(dropout))
    return model


def DenseTop(model, units, dropout):
    
    model.add(Flatten())
    for u in units:
        model.add(Dense(u, activation='relu'))
        if dropout > 0.0:
            model.add(Dropout(dropout))
    # Output 
    model.add(Dense(1, activation="sigmoid"))
    return model


def getNormModel():
    
    model = Sequential()
    
    # Convolution + ReLU + BatchNorm + Maxpool
    model = ConvBlockNorm(model, 32, 1, 0.0,input_shape=(75, 75, 2))
    model = ConvBlockNorm(model, 64, 1, 0.0)
    model = ConvBlockNorm(model, 128, 1, 0.0)
    model = ConvBlockNorm(model, 256, 1, 0.0)
    model = ConvBlockNorm(model, 256, 1, 0.0)
    
    # Top
    model = DenseTop(model, [512, 256], 0.4)
    
    return model
