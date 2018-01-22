# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:21:45 2018

@author: Borislav
"""

from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D,Flatten, Activation, Input, Concatenate, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

#%%

"""
Input 5x5 conv, 3x3 Maxpool
"""

class DenseNet(object):
    
    def __init__(self, input_size, L_block, k, compression, fc_units, fc_drop):
        self.input = None
        self.output = None
        self.model = None
        
        self.L_block = L_block
        
        self.k = k
        self.compression = compression
        
        self.fc_units = fc_units
        self.fc_drop = fc_drop
        
        self.input_size = input_size
        
        
        
    def forward(self):
        self.input = Input(self.input_size)
        
        # Start of the network in a large convolution followed by maxpool
        x = Conv2D(2 * self.k, kernel_size=(5,5), padding='same', activation='relu')(self.input)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
        
        # Stack denseBlocks followed by transition layer
        for L in self.L_block[:-1]:
            x = self.denseBlock(x, self.k, L)
            x = self.transitionLayer(x, self.k, self.compression) 
        x = self.denseBlock(x, self.k, self.L_block[-1])
        # Top layer
        x = Flatten()(x)
        for units in self.fc_units:
            x = Dense(units)(x)
            if self.fc_drop is not None:
                x = Dropout(self.fc_drop)(x)
    
        # Define the model
        self.output = x
        self.model = Model(inputs=self.input, outputs=self.output)
    
    
    def denseBlock(self, input_layer, n_filters, n_layers):
        """
        Dense block - connect every layer to every other layer in the block
        First convolve the input, then convolve again, then concat the input and keep convolving
        """
        nodes = []
        x = Conv2D(4*n_filters, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
        x = Conv2D(n_filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
        nodes.append(x)
        for i in range(n_layers-1):
            if i == 0:
                x = nodes[0]
            else:
                x = Concatenate(axis=-1)(nodes) 
            x = Conv2D(4*n_filters, kernel_size=(1, 1), padding='same', activation='relu')(x)
            x = Conv2D(n_filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
            nodes.append(x)
        return x
    
    def transitionLayer(self, x, n_filters, compression):
        
        filters = int(n_filters*compression)
        x = Conv2D(filters, kernel_size=(1, 1), padding='same', activation='relu')(x)
        x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
        return x
    
    
        
net = DenseNet(input_size=(75,75,2), 
               L_block = [6,6,6,10], 
               k=12, 
               compression=0.5, 
               fc_units=[1024, 1], 
               fc_drop=0.5)
net.forward()
net.model.summary()



#optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)
net.model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])