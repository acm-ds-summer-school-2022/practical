import numpy as np
import pickle
import matplotlib.pyplot as plt
import numpy.ma as ma
import re
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.preprocessing as prep
from sklearn.model_selection import train_test_split
from model import AugementedConvLSTM
import configparser
import argparse
import h5py
import glob
import random
projection_dimensions = [32,32]

#%%
def normalize(data):
    data = data - data.mean()
    data = data / data.std()
    return data

def set_data(X, Y,):
    channel=7
    X_normalized = np.zeros((int(channel), np.max(X.shape), int(projection_dimensions[0]), int(projection_dimensions[1])))
    for i in range(7):
        X_normalized[i,] = normalize(X[i,])
    Y_normalized = normalize(Y)
    std_observed = Y.std()  
    X = X_normalized.transpose(1,2,3,0)
    Y = Y_normalized.reshape(-1,projection_dimensions[0], projection_dimensions[1], 1)
    return X, Y, std_observed

def data_generator(X,Y):
    min_train_year = 1948
    max_train_year = 1999
    min_test_year = 2000
    max_test_year = 2005
    total_years = max_test_year - min_train_year + 1
    train_years = max_train_year - min_train_year + 1
    n_days = np.max(X.shape)
    train_days = int((n_days/total_years)*train_years)
    train_x, train_y = X[:train_days], Y[:train_days]
    test_x, test_y = X[train_days:], Y[train_days:]
    time_steps = 4
    batch_size1 = 15
    train_generator = prep.sequence.TimeseriesGenerator(
        train_x, 
        train_y.reshape(-1, projection_dimensions[0], projection_dimensions[1], 1),
        length=time_steps, 
        batch_size=batch_size1
        )
    test_generator = prep.sequence.TimeseriesGenerator(
        test_x, 
        test_y.reshape(-1, projection_dimensions[0], projection_dimensions[1], 1),
        length=time_steps, 
        batch_size=batch_size1
        )
    return train_generator, test_generator

def train(clstm_model, train_generator, val_generator, load_weights = False, std_observed = 1.0, epochs = 16):
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    def actual_rmse_loss(y_true, y_pred):
        return K.sqrt(K.mean(K.square((y_pred - y_true)*std_observed)))
    adam = tf.keras.optimizers.Adam(learning_rate=0.0003)
    clstm_model.compile(optimizer=adam, loss=root_mean_squared_error, metrics=[root_mean_squared_error, actual_rmse_loss])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f"convlstm_weights_pr_toy.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    termnan = tf.keras.callbacks.TerminateOnNaN()
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_delta=0.005, min_lr=0.000004, verbose=1)
    callbacks_list = [checkpoint, reduce_lr, termnan]
    history = clstm_model.fit(
        train_generator, 
        callbacks=callbacks_list, 
        epochs=epochs, 
        validation_data=val_generator,
        verbose=1
        )
    return history

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s
    
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

#%%
class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, list_examples, batch_size=15, timesteps = 4, n_channels = 7, dim=(50, 50), shuffle=True):
    # Constructor of the data generator.
    self.dim = dim
    self.batch_size = batch_size
    self.timesteps = timesteps
    self.n_channels = n_channels
    self.list_examples = list_examples
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    # Denotes the number of batches per epoch
    return int(np.floor(len(self.list_examples) / (self.timesteps*self.batch_size)))

  def __getitem__(self, index):
    # Generate one batch of data
    indexes = self.indexes[index*self.batch_size*self.timesteps:(index+1)*self.batch_size*self.timesteps]
    # Find list of IDs
    list_IDs_temp = [self.list_examples[k] for k in indexes]
    # Generate data
    assert len(list_IDs_temp) == self.timesteps*self.batch_size
    X, y = self.__data_generation(list_IDs_temp)
    return X, y

  def on_epoch_end(self):
    # This function is called at the end of each epoch.
    self.indexes = np.arange(len(self.list_examples))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, list_IDs_temp):
    # Load individual numpy arrays and aggregate them to a batch.
    X = np.empty([self.batch_size, self.timesteps, self.n_channels, self.dim[0], self.dim[1]], dtype=np.float32)
    y = np.empty([self.batch_size, 1, self.dim[0], self.dim[1]], dtype=np.float32)
    # # Generate data.
    j = 0
    for i, ID in enumerate(list_IDs_temp):
        # Load sample
        X[j,(i)%self.timesteps,:,:,:] = np.load(ID[0])
        # Load labels  
        if (i+1)%self.timesteps == 0 and i != 0:
          y[j,:,:,:] = np.load(ID[1])
          j += 1

    return X.transpose(0,1,3,4,2), y.transpose(0,2,3,1)

 