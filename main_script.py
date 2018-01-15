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
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D,Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

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
X_train.astype(np.float32)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=92)
print("Fraction of positive examples in train {:.4f}".format(np.sum(y_train) / y_train.shape[0]))
print("Fraction of positive examples in valid {:.4f}".format(np.sum(y_val) / y_val.shape[0]))

def normalize(X):
    # Normalize all images in a dataset X, where axis 1 is the image
    # (X - m_X) / std_X, per filter
    X_norm = np.zeros(shape=X.shape)
    for i in range(X.shape[3]):
        X_norm[:,:,:,i] = (X[:,:,:,i] - X[:,:,:,i].mean(axis=1, keepdims=True)) / X[:,:,:,i].std(axis=1, keepdims=True)
    return X_norm

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
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    #Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = getModel()
model.summary()



#%%
"""
Train Model
"""     
batch_size = 256
n_epochs = 50
steps_per_epoch = X_train.shape[0] // batch_size
weights_path="model/weights/weights-{epoch:03d}-{val_acc:.3f}.hdf5" # format to save

# Write the name of the model you want to load
model_path = "model/weights/weights-044-0.869.hdf5"                 

if model_path:
    model.load_weights(model_path)
    print("Model Loaded")

def getCallbacks():
    # Callbacks
    checkpoint = ModelCheckpoint(weights_path, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1,  epsilon=0.001, cooldown=2, min_lr=1e-12)
    tb = TensorBoard(log_dir='./model/log/', histogram_freq=1, batch_size=batch_size, write_graph=False, write_grads=True, write_images=False)
    return [checkpoint, earlyStop, tb, lr_schedule]
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
                    initial_epoch=26) 

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



        
            
