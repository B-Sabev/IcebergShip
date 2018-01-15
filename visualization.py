# -*- coding: utf-8 -*-

#%%
from bcolz import carray as bcolzarray
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

#%%

#Load training data
X_train = np.array(bcolzarray(rootdir='data/processed/train/X', mode='r'))
y_train = np.array(bcolzarray(rootdir='data/processed/train/y', mode='r'))
angle_train = np.array(bcolzarray(rootdir='data/processed/train/angle', mode='r'))

# Load test data
X_test = np.array(bcolzarray(rootdir='data/processed/test/X', mode='r'))
ids = np.array(bcolzarray(rootdir='data/processed/test/ids', mode='r'))
angle_test = np.array(bcolzarray(rootdir='data/processed/test/angle', mode='r'))
is_natural = np.array(bcolzarray(rootdir='data/processed/test/is_natural', mode='r'))

#%%
plt.hist([angle_train, angle_test], 100, stacked=True, label=['train', 'test'])
plt.legend()
plt.title("Incident angle distribution for train and test data")
plt.xlabel("degree of angle")
plt.show()


#%%

"""
Classlabel X Bands

Different bands 

Natural vs Machine-generated

"""



#%%

#TODO consider adding multiplication


def plot_multi(X):
    n_imgs = 10
    f, ax = plt.subplots(n_imgs, 6, figsize=(12, 4 * n_imgs))
    for i in range(n_imgs):
        ax[i][0].imshow(X[i,:,:,0])
        ax[i][0].set_title("Ship" if y_train[i] == 0 else "Iceberg")
        ax[i][0].axis('off')
        
        ax[i][1].imshow(X[i,:,:,1])
        ax[i][1].axis('off')
    
        ax[i][2].imshow(X[i,:,:,1] * X[i,:,:,0])
        ax[i][2].axis('off')
        
        ax[i][3].imshow(X[i,:,:,0] - X[i,:,:,1])
        ax[i][3].axis('off')
        
        ax[i][4].imshow(X[i,:,:,1] + X[i,:,:,0])
        ax[i][4].axis('off')
        
        
        ax[i][5].imshow(np.maximum(X[i,:,:,1], X[i,:,:,0]))
        ax[i][5].axis('off')
        
        
        
        
        
        
def normalize(X):
    # Normalize all images in a dataset X, where axis 1 is the image
    # (X - m_X) / std_X, per filter
    X_norm = np.zeros(shape=X.shape)
    for i in range(X.shape[2]):
        X_norm[:,:,i] = (X[:,:,i] - X[:,:,i].mean(axis=1, keepdims=True)) / X[:,:,i].std(axis=1, keepdims=True)
    return X_norm

# First normalize, then reshape to proper format
X_train_norm = normalize(X_train)
X_train_norm = np.reshape(X_train_norm, (-1, 75,75,2))

X_train = np.reshape(X_train, (-1, 75,75,2))



plot_multi(X_train)

print("\n\n\n\n\n\n\n")

plot_multi(X_train_norm)



    
   


#%%
"""
Plot angles and random images of the 2 bands for each of the classes
"""
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
            
image_class_gallery(X_train_norm, y_train, n_imgs=6) 
