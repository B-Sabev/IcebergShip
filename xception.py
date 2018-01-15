# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:38:47 2018

@author: Borislav
"""

#%%
"""
Xception
"""
from keras.applications.xception import Xception
from keras.layers import Input


def getXceptionImageModel():
    # Base Xception model
    base_model = Xception(include_top=True, weights=None, input_tensor=Input(shape=(75, 75, 2)))
    # output of the model is 1000 units
    x = base_model.output
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = getXceptionImageModel()

model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])