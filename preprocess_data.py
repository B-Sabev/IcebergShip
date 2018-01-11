# -*- coding: utf-8 -*-
"""
The script needs the original Kaggle files in data/raw folder

This loads the JSON-s of the data and saves nessesary data as bcolz

X - float32, band1 and band2 concatenated
angle - angle at which the images are taken
ids - unique id of every picture
labels - ground-truth labels ship or iceberg (only for train data)

is_natural - additonal variable not contained in the dataset
            1 if the image is natural, 0 if it is machine generated (only for test data)
"""


#%%

import numpy as np
import bcolz
import json
from time import time

#%%
# Load data

with open("data/raw/train.json", 'r') as f:
        train_data = json.load(f)
with open("data/raw/test.json", 'r') as f:
        test_data = json.load(f)       

# First 1604 are train, next are test      
data = train_data + test_data

#%%
# convert to numpy array

angle = np.array([dat['inc_angle'] for dat in data])
ids = np.array([dat['id'] for dat in data])
band_1 = np.array([dat['band_1'] for dat in data])
band_2 = np.array([dat['band_2'] for dat in data])
label = np.array([dat['is_iceberg'] for dat in data[:1604]])

#%%
# In natural images, angle has 4 point percison, while machine generated has 8 or more
ang_len = np.array([len(a) for a in angle])
is_natural = (ang_len <= 7).astype(int)

#%%

# replace missing values of angle with -1, convert to float32
angle = np.where(angle == 'na', '-1', angle)
angle = angle.astype(np.float32)


#%%

# Stack the 2 bands in 2 channel image across the 3rd axis
X = np.stack((band_1, band_2), axis=2)
X = X.astype(np.float32)

#%%

print(sum(is_natural[:1604]))
# train set has only natural images, so exclude is_natural for train
is_natural = is_natural[1604:]

#%%
"""
All predictors are float32,
labels are int32, 
ids are string

Save as bcolz for fast loading
"""

# Train
train_dir = "data/processed/train/"

X_train = bcolz.carray(X[:1604, :, :], rootdir=train_dir + 'X')
X_train.flush()

angle_train = bcolz.carray(angle[:1604], rootdir=train_dir + 'angle')
angle_train.flush()

ids_train = bcolz.carray(ids[:1604], rootdir=train_dir + 'ids')
ids_train.flush()

labels = bcolz.carray(label, rootdir=train_dir + 'y')
labels.flush()

# Test

test_dir = "data/processed/test/"

X_test = bcolz.carray(X[1604:, :, :], rootdir=test_dir + 'X')
X_test.flush()

angle_test = bcolz.carray(angle[1604:], rootdir=test_dir + 'angle')
angle_test.flush()

ids_test = bcolz.carray(ids[1604:], rootdir=test_dir + 'ids')
ids_test.flush()

is_natural = bcolz.carray(is_natural, rootdir=test_dir + 'is_natural')
is_natural.flush()

