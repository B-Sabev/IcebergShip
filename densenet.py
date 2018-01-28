# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:34:50 2018

@author: Borislav
"""

from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D,Flatten, Activation, Input, Concatenate, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau


#%%

"""
Define the densenet class
"""
class DenseNet(object):
    
    def __init__(self, input_size, L, k, 
                 compression=1, 
                 use_bottleneck=False, 
                 dropout_rate=None, 
                 n_dense_blocks=3,
                 strides=(1,1)):
        self.input = None
        self.output = None
        self.L = L
        self.k = k
        self.use_bottleneck = use_bottleneck
        self.compression = compression
        self.dropout_rate = dropout_rate
        self.n_dense_blocks = n_dense_blocks
        self.input_size = input_size
        self.strides = strides

    def getModel(self):
        self.forward()
        return Model(inputs=self.input, outputs=self.output)
    
    def forward(self):
        self.input = Input(self.input_size)
        
        # Start of the network in a large convolution followed by maxpool
        
        x = Conv2D(2 * self.k, kernel_size=(5,5), strides=self.strides, padding='same')(self.input)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
        
        # 4 - Conv2d at the start, 1 dense at the end, 2 transitional
        # Divided by 3 because we have 3 dense blocks with 1 conv2d per layer
        #         or 6 because 3 dense blocks with 2 conv2d layers when use_bottleneck
        if self.use_bottleneck:
            n_layers = (self.L - 4) // (2*self.n_dense_blocks)
        else:
            n_layers = (self.L - 4) // self.n_dense_blocks
        
        # Stack denseBlocks followed by transition layer
        for i in range(self.n_dense_blocks-1):
            x = self.denseBlock(x, self.k, n_layers)
            x = self.transitionLayer(x, self.k, self.compression) 
        x = self.denseBlock(x, self.k, n_layers)
        x = AveragePooling2D(pool_size=(2,2))(x)
        x = Activation('relu')(x)
        
        # Top layer
        x = Flatten()(x)
        if self.dropout_rate is not None:
                x = Dropout(self.dropout_rate)(x)
        # Output
        self.output = Dense(1, activation='sigmoid')(x)
    
    def denseBlock(self, input_layer, n_filters, n_layers):
        """
        Dense block - connect every layer to every other layer in the block
        First convolve the input, then convolve again, then concat the input and keep convolving
        """
        nodes = []
        if self.use_bottleneck:
            x = self.bottleneck(input_layer, n_filters)
        else:
            x = self.plainLayer(input_layer, n_filters)
        nodes.append(x)
        for i in range(n_layers-1):
            if i == 0:
                x = nodes[0]
            else:
                x = Concatenate(axis=-1)(nodes) 
            # use bottleneck layer or not   
            if self.use_bottleneck:
                x = self.bottleneck(x, n_filters)
            else:
                x = self.plainLayer(x, n_filters)
            nodes.append(x)
        return x
    
    def plainLayer(self, input_layer, n_filters):
        x = BatchNormalization()(input_layer)
        x = Activation('relu')(x)
        x = Conv2D(n_filters, kernel_size=(3, 3), padding='same')(x)
        return x
        
    def bottleneck(self, input_layer, n_filters):
         #BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)
         x = BatchNormalization()(input_layer)
         x = Activation('relu')(x)
         x = Conv2D(4*n_filters, kernel_size=(1, 1))(x)
         x = BatchNormalization()(x)
         x = Activation('relu')(x)
         x = Conv2D(n_filters, kernel_size=(3, 3), padding='same')(x)
         return x
        
    def transitionLayer(self, x, n_filters, compression):
        """
        Bottleneck the features discovered by DenseBlock with compression param,
        AveragePool over the output
        """
        filters = int(n_filters*compression)
        x = Conv2D(filters, kernel_size=(1, 1), activation='relu')(x)
        x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
        return x
    

