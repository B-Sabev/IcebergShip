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
        x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
        # Top layer
        x = Flatten()(x)
        for units in self.fc_units:
            x = Dense(units, activation='relu')(x)
            if self.fc_drop is not None:
                x = Dropout(self.fc_drop)(x)
        
    
        # Define the model
        self.output = Dense(1, activation='sigmoid')(x)
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
        """
        Bottleneck the features discovered by DenseBlock with compression param,
        AveragePool over the output
        """
        filters = int(n_filters*compression)
        x = Conv2D(filters, kernel_size=(1, 1), padding='same', activation='relu')(x)
        x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
        return x
    
net = DenseNet(input_size=(75,75,2), 
               L_block = [6,6,6], 
               k=40, 
               compression=0.5, 
               fc_units=[512], 
               fc_drop=0.5)
net.forward()
net.model.summary()


#%%

    earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1,  epsilon=0.001, cooldown=1, min_lr=1e-12)
    #tb = TensorBoard(log_dir='./model/log/', histogram_freq=1, batch_size=batch_size, write_graph=False, write_grads=True, write_images=False)
    return [checkpoint,earlyStop,lr_schedule]


#%%
"""
Train on small batches
"""
batch_size = 64
n_epochs = 100
steps_per_epoch = 1#X_train.shape[0] // batch_size
weights_path="model/weights/short_epochs/weights-{epoch:03d}-{val_acc:.3f}.hdf5" # format to save

# Write the name of the model you want to load
model_path = None              

if model_path:
    #model.load_weights(model_path)
    print("Model Loaded")

checkpoint = ModelCheckpoint(weights_path, monitor='val_acc', verbose=2, save_best_only=False, mode='max')
callbacks_list = [checkpoint]

train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_gen = val_datagen.flow(X_val, y_val, batch_size=8) # validate only on 128 examples 


# Fit 1 batch for enough epochs to iterate over all data
model.fit_generator(train_gen,
                    steps_per_epoch = 1,
                    epochs = X_train.shape[0] // batch_size,
                    verbose = 2) 
# 
model.fit_generator(train_gen,
                    steps_per_epoch = 0,
                    epochs = 1,
                    callbacks = callbacks_list,
                    validation_data=val_gen,
                    verbose = 2)


