# -*- coding: utf-8 -*-

#%%
# Imports 
from bcolz import carray as bcolzarray
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D,Flatten
from keras.callbacks import ModelCheckpoint

#%%

# Load all train data
X_train = np.array(bcolzarray(rootdir='data/processed/train/X', mode='r'))
y_train = np.array(bcolzarray(rootdir='data/processed/train/y', mode='r'))
angle_train = np.array(bcolzarray(rootdir='data/processed/train/a', mode='r'))
#%%
"""
Plot angles and random images of the 2 bands for each of the classes
"""

plt.hist(angle_train)
plt.title("Distribution of angles")
plt.show()
np.random.seed(505)
def image_class_gallery(X, y, n_imgs=5):
    select0 = np.random.choice(np.squeeze(np.argwhere(y==0)), n_imgs)
    select1 = np.random.choice(np.squeeze(np.argwhere(y==1)), n_imgs)
    band_1 = np.squeeze(X[:,:,0])
    band_2 = np.squeeze(X[:,:,1])
    imgs = [ 
            band_1[select0,],
            band_2[select0,],
            band_1[select1,],
            band_2[select1,]
           ]
    f, ax = plt.subplots(n_imgs, 4, figsize=(20, 6 * n_imgs))
    for row in range(n_imgs):
        for im in range(4):
            current_img = np.reshape(imgs[im][row,], (75,75))
            ax[row][im].imshow(current_img)
            title = "Ship" if im < 2 else "Iceberg"
            title += " HH" if im==0 or im==2 else " HV"
            ax[row][im].set_title(title)
            ax[row][im].axis('off')
            
image_class_gallery(X_train, y_train, n_imgs=6) 

#%%
"""
Split into train and validation dataset
"""
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=92)
print("Fraction of positive examples in train {:.4f}".format(np.sum(y_train) / y_train.shape[0]))
print("Fraction of positive examples in valid {:.4f}".format(np.sum(y_val) / y_val.shape[0]))

#%%
"""
Define basic keras model
"""

model = Sequential() 
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(75, 75, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  


model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

#%%
"""
Image augmentation

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


def normalize(X):
    """
    Normalize all images in a dataset X, where axis 1 is the image
    """
    return (X - X.mean(axis=1, keepdims=True))/ X.std(axis=1, keepdims=True) 
X_train = normalize(X_train)
X_val = normalize(X_val)

# Reshape to image format
X_train = np.reshape(X_train, (-1, 75,75,2))
X_val = np.reshape(X_val, (-1, 75,75,2))

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_gen = train_datagen.flow(X_train, y_train, batch_size=128)
val_gen = val_datagen.flow(X_val, y_val, batch_size=256)    

#%%     
filepath="model/weights-{epoch:03d}-{val_acc:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# Fit model
model.fit_generator(train_gen,
                    steps_per_epoch = 2,
                    epochs = 5,
                    callbacks = callbacks_list,
                    validation_data=val_gen,
                    verbose = 2,
                    validation_steps=1,
                    initial_epoch=2)            
            
            
