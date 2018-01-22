# -*- coding: utf-8 -*-

#%%
"""
Add all nessesary imports 
"""
from bcolz import carray as bcolzarray
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D,Flatten, Activation, Input,  Concatenate, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam



#%%
"""
Load training data
"""
X_train = np.array(bcolzarray(rootdir='data/processed/train/X', mode='r'))
y_train = np.array(bcolzarray(rootdir='data/processed/train/y', mode='r'))
angle_train = np.array(bcolzarray(rootdir='data/processed/train/angle', mode='r'))

# TODO - try with a better 3rd channel
# Add third channel as sum of the 2 channels
#X_3rd = (X_train[:, :, 0] + X_train[:, :, 1]).reshape(X_train.shape[0], -1, 1)
#X_train = np.append(X_train, X_3rd, axis=2)


X_train = X_train.reshape(-1, 75, 75, 2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=92)
print("Fraction of positive examples in train {:.4f}".format(np.sum(y_train) / y_train.shape[0]))
print("Fraction of positive examples in valid {:.4f}".format(np.sum(y_val) / y_val.shape[0]))

def normalize(X):
    # Normalize all images in a dataset X, where axis 1 is the image
    # (X - m_X) / std_X, per filter
    X_norm = np.zeros(shape=X.shape)
    for i in range(X.shape[3]):
        X_norm[:,:,:,i] = (X[:,:,:,i] - X[:,:,:,i].mean(keepdims=True)) / X[:,:,:,i].std(keepdims=True)
    return X_norm.astype(np.float32)

# Normalization 0 mean, 1 std
X_train = normalize(X_train)
X_val = normalize(X_val)


#%%
"""
Image augmentation
"""

train_datagen = ImageDataGenerator(
                                    rotation_range=0.3,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True
                                    )


#%%

"""
Define the densenet class
"""
class DenseNet(object):
    
    def __init__(self, input_size, L_block, k, compression, fc_units, fc_drop):
        self.input = None
        self.output = None
        self.L_block = L_block
        self.k = k
        self.compression = compression
        self.fc_units = fc_units
        self.fc_drop = fc_drop
        self.input_size = input_size

    
    def getModel(self):
        return Model(inputs=self.input, outputs=self.output)
    
    def forward(self):
        self.input = Input(self.input_size)
        
        # Start of the network in a large convolution followed by maxpool
        x = Conv2D(2 * self.k, kernel_size=(5,5), padding='same')(self.input)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
        
        # Stack denseBlocks followed by transition layer
        for L in self.L_block[:-1]:
            x = self.denseBlock(x, self.k, L)
            x = self.transitionLayer(x, self.k, self.compression) 
        x = self.denseBlock(x, self.k, self.L_block[-1])
        x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
        x = Activation('relu')(x)
        # Top layer
        x = Flatten()(x)
        for units in self.fc_units:
            x = Dense(units, activation='relu')(x)
            if self.fc_drop is not None:
                x = Dropout(self.fc_drop)(x)
        
    
        # Define the model
        self.output = Dense(1, activation='sigmoid')(x)
    
    def denseBlock(self, input_layer, n_filters, n_layers):
        """
        Dense block - connect every layer to every other layer in the block
        First convolve the input, then convolve again, then concat the input and keep convolving
        """
        nodes = []
        x = self.bottleneck(input_layer, n_filters)
        nodes.append(x)
        for i in range(n_layers-1):
            if i == 0:
                x = nodes[0]
            else:
                x = Concatenate(axis=-1)(nodes) 
            x = self.bottleneck(x, n_filters)
            nodes.append(x)
        return x
    
    def bottleneck(self, input_layer, n_filters):
         #BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)
         x = BatchNormalization()(input_layer)
         x = Activation('relu')(x)
         x = Conv2D(4*n_filters, kernel_size=(1, 1), padding='same')(x)
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
        x = Conv2D(filters, kernel_size=(1, 1), padding='same', activation='relu')(x)
        x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
        return x


net = DenseNet(input_size=(75,75,2), 
               L_block = [12,12,12], 
               k=24, 
               compression=0.5, 
               fc_units=[1024, 1], 
               fc_drop=0.5)

net.forward()
model = net.getModel()
model.summary()

#%%

loss = 'binary_crossentropy'        
optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)
metrics = ['accuracy']
    


#net = DenseNet(input_size=(75,75,2), L_block = [6,6,6,10], k=12, compression=0.5, fc_units=[1024, 1], fc_drop=0.5)
#net.forward()
model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)


#%%
"""
Train Model
"""     
batch_size = 32
n_epochs = 20
steps_per_epoch = X_train.shape[0] // batch_size
weights_path="model/weights/weights-{epoch:03d}-{val_acc:.3f}.hdf5" # format to save

# Write the name of the model you want to load
model_path = None              

if model_path:
    #model.load_weights(model_path)
    print("Model Loaded")

def getCallbacks():
    # Callbacks
    checkpoint = ModelCheckpoint(weights_path, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1,  epsilon=0.001, cooldown=1, min_lr=1e-12)
    #tb = TensorBoard(log_dir='./model/log/', histogram_freq=1, batch_size=batch_size, write_graph=False, write_grads=True, write_images=False)
    return [checkpoint,earlyStop,lr_schedule]
callbacks_list = getCallbacks()
print("Callbacks ready")
# Train data 
train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size)


print("Fitting model")
# Fit model
model.fit_generator(train_gen,
                    steps_per_epoch = steps_per_epoch,
                    epochs = n_epochs,
                    callbacks = callbacks_list,
                    validation_data=(X_val, y_val),
                    verbose = 2,
                    initial_epoch=0) 

print("Done training")

"""
Only after happy with the validation accuracy continue forward
"""

#%%

"""
PseudoLabeling   - try with and without  
"""


"""
Fit the validation data before the final test - train + 5*validation data for 3 epochs
"""
# Stack Train and 5 times validation data
X_train_on_val = np.vstack([X_train, X_val, X_val, X_val, X_val, X_val])
y_train_on_val = np.hstack([y_train, y_val, y_val, y_val, y_val, y_val])

def shuffle_2_arrays(X,y):
    p = np.random.permutation(X.shape[0])
    X_shuffled = X[p,:,:,:]
    y_shuffled = y[p]
    return (X_shuffled, y_shuffled)

X_train_on_val, y_train_on_val = shuffle_2_arrays(X_train_on_val,y_train_on_val)

train_gen = train_datagen.flow(X_train_on_val, y_train_on_val, batch_size=128)

model.fit_generator(train_gen,
                    steps_per_epoch = 25,
                    epochs = 3,
                    callbacks = callbacks_list,
                    verbose = 2) 


#%%

# Load test data
X_test = np.array(bcolzarray(rootdir='data/processed/test/X', mode='r'))
ids = np.array(bcolzarray(rootdir='data/processed/test/ids', mode='r'))
#angle_test = np.array(bcolzarray(rootdir='data/processed/test/a', mode='r'))
is_natural = np.array(bcolzarray(rootdir='data/processed/test/is_natural', mode='r'))


#%%
"""
Predict on test data and write submission
"""    
# Pick a name for the submission file
sub_name = "Submission44"


# Normalize image and put it in the proper shape
X_test = np.reshape(X_test, (-1,75,75,2))
X_test = normalize(X_test)


predictions = model.predict(X_test, verbose=1)
"""
Clipping or raising the predictions to power to soften them
"""

#predictions = np.clip(predictions ** 1.6, 0.03, 0.97)

# write submission to csv
submission = pd.DataFrame()
submission['id']=ids
submission['is_iceberg']=predictions.reshape(-1)
submission.to_csv('submissions/{}.csv'.format(sub_name), index=False)    



#%%



        
            
