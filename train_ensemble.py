# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:54:59 2018

@author: Borislav
"""

from bcolz import carray as bcolzarray
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from densenet import DenseNet

#%%

"""
Load training data
"""
X = np.array(bcolzarray(rootdir='data/processed/train/X', mode='r'))
y = np.array(bcolzarray(rootdir='data/processed/train/y', mode='r'))

def norm_data(X_train, X_val):
    """
    Featurewise normalization, X_val uses the features for X_train
    Assumed 3d arrays, normalize the 2 channels seperatly
    """
    # Compute mean and std for each of the bands
    band1_mean = X_train[:,:,0].mean(axis=0)
    band1_std = X_train[:,:,0].std(axis=0)
    band2_mean = X_train[:,:,1].mean(axis=0)
    band2_std = X_train[:,:,1].std(axis=0)
    # Compute the new X_train and X_val 
    X_train_norm = np.zeros(X_train.shape)
    X_train_norm[:,:,0] = (X_train[:,:,0] - band1_mean) / band1_std
    X_train_norm[:,:,1] = (X_train[:,:,1] - band2_mean) / band2_std
    
    X_val_norm = np.zeros(X_val.shape)
    X_val_norm[:,:,0] = (X_val[:,:,0] - band1_mean) / band1_std
    X_val_norm[:,:,1] = (X_val[:,:,1] - band2_mean) / band2_std
    return X_train_norm, X_val_norm


def plot_accuracy(history):
    # Plot train and validation accuracy
    plt.plot(history.history['acc'], color='blue', label='Train accuracy')
    plt.plot(history.history['val_acc'], color='red', label='Validation accuracy')
    plt.xlabel("Epochs")
    plt.title("Train and validation accuracy during training for {}".format(name))
    plt.legend()
    plt.ylim([-0.01, 1.01])
    plt.show()
    
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
Make all models that qualify for the ensembler
"""

# Depth and growth factor of the networks
Ls = [28, 40, 24, 34, 32]
ks = [12, 16, 12, 16, 16]
nums_dense_blocks = [3, 2, 2, 3, 2]

DenseNets = {}

# Compile options
loss = 'binary_crossentropy'        
optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
metrics = ['accuracy']

for L,k, n_dense_blocks in zip(Ls,ks, nums_dense_blocks ):
    name = "L={}_k={}_{}".format(L,k, n_dense_blocks)
    temp_net = DenseNet(input_size=(75,75,2), 
                        L=L, 
                        k=k, 
                        compression=0.5, 
                        use_bottleneck=True,
                        dropout_rate=None,
                        n_dense_blocks=n_dense_blocks,
                        strides=(1,1))
    
    DenseNets[name] = temp_net.getModel()
    DenseNets[name].compile(loss=loss,
                            optimizer=optimizer,
                            metrics=metrics)
    
for name, model in DenseNets.items():
    print("Model {} has {} parameters".format(name, model.count_params()))    
 

#%%
"""
Train all models
"""
    
names = ['L=28_k=12_3', 'L=40_k=16_2', 'L=24_k=12_2', 'L=34_k=16_3', 'L=32_k=16_2']
n_model = 0  
# for loading already trained models, turned off for now
dirs = 'model/weights/ensemble/'
paths = {'L=28_k=12_3' : '0_split_DenseNet_L=28_k=12_3-060-0.820.hdf5',
         'L=40_k=16_2' : '1_split_DenseNet_L=40_k=16_2-060-0.854.hdf5',
         'L=24_k=12_2' : '2_split_DenseNet_L=24_k=12_2-060-0.850.hdf5',
         'L=34_k=16_3' : '3_split_DenseNet_L=34_k=16_3-060-0.847.hdf5',
         'L=32_k=16_2' : '4_split_DenseNet_L=32_k=16_2-056-0.797.hdf5'} 

# train setting
batch_size = 64
n_epochs = 60
initial_epoch = 0
steps_per_epoch = X.shape[0] // batch_size    

histories = []
# Define a stratified split into 5 parts
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for n_split, (train_index, val_index) in enumerate(skf.split(X,y)):
    # Split train and test data
    X_train, X_val = X[train_index, :, :], X[val_index, :, :]
    y_train, y_val = y[train_index], y[val_index]
    # Normalize
    X_train, X_val = norm_data(X_train, X_val)
    # Reshape after normalization
    X_train = X_train.reshape(-1, 75, 75, 2)
    X_val = X_val.reshape(-1, 75, 75, 2)
    
    # Select a model and load its weight it they exist
    name = names[n_model]
    model = DenseNets[name]
    model_path = None#dirs + paths[name]           
    if model_path:
        model.load_weights(model_path)
        print("Model Loaded")
    n_model += 1

    # save the weights
    weights_path="model/weights/ensemble/" + str(n_split) + "_split_DenseNet_"+name+"-{epoch:03d}-{val_acc:.3f}.hdf5" # format to save
    
    # Callbacks
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1,  epsilon=0.001, cooldown=1, min_lr=1e-12)    
    checkpoint = ModelCheckpoint(weights_path, monitor='val_acc', verbose=2, save_best_only=False, mode='max')
    callbacks_list = [checkpoint, lr_schedule]
    print("Callbacks ready")
    
    # Train data 
    train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    
    print("Fitting model - DenseNetBC {}".format(name))
    # Fit model
    history = model.fit_generator(train_gen,
                        steps_per_epoch = steps_per_epoch,
                        epochs = n_epochs,
                        callbacks = callbacks_list,
                        validation_data=(X_val, y_val),
                        shuffle=True,
                        verbose = 2,
                        initial_epoch=initial_epoch)
    
    # Show results while training and save them for later plotting
    plot_accuracy(history)
    histories.append(history)



#%%
"""
Plot validation accuracy for all models
"""
    
names_for_plot = ['L=28 k=12 #DB=3',
                  'L=40 k=16 #DB=2',
                  'L=24 k=12 #DB=2',
                  'L=34 k=16 #DB=3',
                  'L=32 k=16 #DB=2']

for hist, name in zip(histories, names_for_plot):
    val_loss = hist.history['val_acc']
    plt.plot(val_loss, label=name)
    
plt.title("Validation accuracy of the 5 models across 60 epochs")    
plt.ylabel("Validation accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.ylim([-0.01, 1.01])
plt.show()



#%%
"""
After training: 
Load test data and best models 
"""

# Load test data
X_test = np.array(bcolzarray(rootdir='data/processed/test/X', mode='r'))
ids = np.array(bcolzarray(rootdir='data/processed/test/ids', mode='r'))

# Normalize image and put it in the proper shape
_, X_test = norm_data(X, X_test)
X_test = np.reshape(X_test, (-1,75,75,2))

# Directory to the weights and names of the models
dirs = 'model/weights/ensemble/'
paths = {'L=28_k=12_3' : '0_split_DenseNet_L=28_k=12_3-060-0.820.hdf5',
         'L=40_k=16_2' : '1_split_DenseNet_L=40_k=16_2-060-0.854.hdf5',
         'L=24_k=12_2' : '2_split_DenseNet_L=24_k=12_2-060-0.850.hdf5',
         'L=34_k=16_3' : '3_split_DenseNet_L=34_k=16_3-060-0.847.hdf5',
         'L=32_k=16_2' : '4_split_DenseNet_L=32_k=16_2-056-0.797.hdf5'}

# Load weights
for i, (name, model) in enumerate(DenseNets.items()):
    model.load_weights(dirs + paths[name])
    
    
#%%
"""
Optional PseudoLabeling - try with and without
"""

# Subset only natural images, shuffle them
is_natural = np.array(bcolzarray(rootdir='data/processed/test/is_natural', mode='r'))
X_test_natural = X_test[is_natural.astype(np.bool),:,:,:]
X_test_natural = X_test_natural[
                        np.random.permutation(X_test_natural.shape[0]), :, :, :]

n_epochs_PL = 5

names = ['L=28_k=12_3', 'L=40_k=16_2','L=24_k=12_2','L=34_k=16_3','L=32_k=16_2']
num_model = 0

for train_index, val_index in skf.split(X,y):
    
    # Split train and test data
    X_train, X_val = X[train_index, :, :], X[val_index, :, :]
    y_train, y_val = y[train_index], y[val_index]
    # Normalize
    X_train, X_val = norm_data(X_train, X_val)
    # Reshape after normalization
    X_train = X_train.reshape(-1, 75, 75, 2)
    X_val = X_val.reshape(-1, 75, 75, 2)
    
    
    name = names[num_model]
    model = DenseNets[name]
    num_model += 1
    
    
    # Save the pseudolearning models in different directory
    weights_path = "model/weights/pl/PL_DenseNet_"+name + "-{epoch:03d}-{val_acc:.3f}.hdf5" 
    checkpoint = ModelCheckpoint(weights_path, monitor='val_acc', verbose=2, save_best_only=False, mode='max')
    
    for _ in range(n_epochs_PL):
        # On every epoch, iterate over splits, predict and train
        
        # predict
        y_pseudo = model.predict(X_test_natural)
        y_pseudo = np.squeeze(y_pseudo)
            
        # Use only the once that the model is sure about
        confidence = 0.95
        pred_confident = np.logical_or(y_pseudo > confidence, y_pseudo < 1-confidence)
        X_pseudo = X_test_natural[pred_confident, :, :, :]
        y_pseudo = y_pseudo[pred_confident]
        y_pseudo = np.where(y_pseudo > 0.5, 1, 0)
        print("Examples used {}".format(y_pseudo.shape[0]))
            
        # append with training data
        X_PL = np.append(X_train, X_pseudo, axis=0)
        y_PL = np.append(y_train, y_pseudo)
            
        # Def generator 
        pseudo_gen = train_datagen.flow(X_PL, 
                                            y_PL, 
                                            batch_size=128)
            
        # Fit model
        history = model.fit_generator(pseudo_gen,
                                            steps_per_epoch = len(pseudo_gen),
                                            epochs = 1,
                                            callbacks = [checkpoint],
                                            validation_data=(X_val, y_val),
                                            shuffle=True,
                                            verbose = 2,
                                            initial_epoch=0)

    

#%%
"""
Predict on test data and write submission, re-run after pseudolabeling
"""    
pred = []

for name, model in DenseNets.items():

    # make predictions
    predictions = model.predict(X_test, verbose=1)
    pred.append(predictions) # for latter exploration
    
    # write a sumbission with the name of the model
    submission = pd.DataFrame()
    submission['id']=ids
    submission['is_iceberg']=predictions.reshape(-1)
    submission.to_csv('submissions/{}.csv'.format(name), index=False)
        
# Mean all predictions made from the models
pred_ensemble = np.mean(np.array(pred), axis=0)

submission = pd.DataFrame()
submission['id']=ids
submission['is_iceberg']=pred_ensemble.reshape(-1)
submission.to_csv('submissions/Ensemble.csv', index=False)

#%%

# Top 3 ensemble
pred_40 = pd.read_csv('submissions/L=40_k=16_2.csv').as_matrix()
pred_34 = pd.read_csv('submissions/L=34_k=16_3.csv').as_matrix()
pred_24 = pd.read_csv('submissions/L=24_k=12_2.csv').as_matrix()

y_ = (pred_40[:,1] + pred_34[:,1] + pred_24[:,1]) / 3.0

submission = pd.DataFrame()
submission['id']=ids
submission['is_iceberg']=y_
submission.to_csv('submissions/Ensemble_top3.csv', index=False)



