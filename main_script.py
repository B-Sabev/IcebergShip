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

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D,Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

#%%
"""
Load training data
"""
X_train = np.array(bcolzarray(rootdir='data/processed/train/X', mode='r'))
y_train = np.array(bcolzarray(rootdir='data/processed/train/y', mode='r'))
angle_train = np.array(bcolzarray(rootdir='data/processed/train/a', mode='r'))



#%%
"""
Train and validation set
Image Augmentation
"""

# TODO - try with a third channel

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=92)
print("Fraction of positive examples in train {:.4f}".format(np.sum(y_train) / y_train.shape[0]))
print("Fraction of positive examples in valid {:.4f}".format(np.sum(y_val) / y_val.shape[0]))

def normalize(X):
    # Normalize all images in a dataset X, where axis 1 is the image
    # (X - m_X) / std_X, per filter
    X_norm = np.zeros(shape=X.shape)
    for i in range(X.shape[2]):
        X_norm[:,:,i] = (X[:,:,i] - X[:,:,i].mean(axis=1, keepdims=True)) / X[:,:,i].std(axis=1, keepdims=True)
    return X_norm

# First normalize, then reshape to proper format
X_train = normalize(X_train)
X_train = np.reshape(X_train, (-1,75,75,2))
X_val = normalize(X_val)
X_val = np.reshape(X_val, (-1,75,75,2))

"""
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)
"""

train_datagen = ImageDataGenerator(
                                    rotation_range=0.3,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    horizontal_flip=False,
                                    vertical_flip=False
                                    )
#%%
"""
Define the model

1 pre-build
1 custom with angle
"""

# TODO try other activations

model = Sequential() 


model.add(Conv2D(64, kernel_size=(3,3), input_shape=(75,75,2)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))



model.add(Conv2D(128, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(128, kernel_size=(1,1)))

model.add(Conv2D(128, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(1,1)))

model.add(Conv2D(64, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())


model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.3))


model.add(Dense(1, activation='sigmoid'))  

model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])


#%%
"""
Train Model
"""     
batch_size = 16
n_epochs = 1
n_train_samples = X_train.shape[0]
weights_path="model/weights/weights-{epoch:03d}-{val_acc:.3f}.hdf5"
model_path = None

if model_path:
    model.load_weights(model_path)

# Callbacks
checkpoint = ModelCheckpoint(weights_path, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=15, verbose=1)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1,  epsilon=0.001, cooldown=3, min_lr=1e-12)
# graph is total nonsense
tb = TensorBoard(log_dir='./model/log/', histogram_freq=1, batch_size=batch_size, 
                 write_graph=False, write_grads=True, write_images=False)
callbacks_list = [checkpoint, earlyStop, tb, lr_schedule]


train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size)


# Fit model
model.fit_generator(train_gen,
                    steps_per_epoch = n_train_samples // batch_size - 8,
                    epochs = n_epochs,
                    callbacks = callbacks_list,
                    validation_data=(X_val, y_val),
                    verbose = 2,
                    validation_steps=1,
                    initial_epoch=0) 



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
"""
Predict on test data and write submission
"""    
# Pick a name for the submission file
sub_name = "Submission"

# Load test data
X_test = np.array(bcolzarray(rootdir='data/processed/test/X', mode='r'))
ids = np.array(bcolzarray(rootdir='data/processed/test/ids', mode='r'))
angle_test = np.array(bcolzarray(rootdir='data/processed/test/a', mode='r'))

# Normalize image and put it in the proper shape
X_test = normalize(X_test)
X_test = np.reshape(X_test, (-1,75,75,2))

predictions = model.predict(X_test, verbose=1)
# write submission to csv
submission = pd.DataFrame()
submission['id']=ids
submission['is_iceberg']=predictions.reshape(-1)
submission.to_csv('submissions/{}.csv'.format(sub_name), index=False)            
            
