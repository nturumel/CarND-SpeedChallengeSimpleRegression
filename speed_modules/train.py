from globals import  global_var

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Activation
from keras import activations
from keras.callbacks import ModelCheckpoint, TensorBoard, History, EarlyStopping 
import keras.metrics
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

import numpy as np
import csv
import datetime
import os

def get_model():
    model = Sequential()
    model.add(Input(shape = (global_var['INPUT_SHAPE'])))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['MeanSquaredError'])
    return model

def train():
    # initialise
    train_input = global_var['SAVE_ARRAY_FILE']
    train_output = global_var['TRAIN_OUTPUT']
    
    # split into input (X) and output (Y) variables
    X = np.load(train_input)
    Y = np.loadtxt(train_output)
    Y = Y[:-1]

    # train the model
    model = get_model()
    print(model.summary()
    )
    # split 
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=1)

    scaler= StandardScaler()
    print((X_train.reshape(-1, X_train.shape[-1])).shape)
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    print(X_train.shape)

    Y_train /= global_var['SCALE_OUTPUT']
    Y_val /= global_var['SCALE_OUTPUT']

    '''
    scaler_Y = StandardScaler()
    print(Y_train.shape)
    Y_train = scaler_Y.fit_transform(Y_train)
    Y_val = scaler_Y.transform(Y_val)
    print("Final scaler Y: ", scaler_Y)
    '''
    # callbacks
    checkpoint = ModelCheckpoint(global_var['MODEL_PATH'], verbose=1, monitor='val_accuracy', save_best=True)
    logdir = os.path.join(global_var['LOG_DIR'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True, update_freq=1)
    history = History()
    callbacks_list = [checkpoint, tensorboard, history]

   # run
    model.fit(X_train, Y_train, batch_size=global_var['BATCH_SIZE'], epochs=global_var['EPOCHS'],  validation_data=(X_val, Y_val), callbacks=callbacks_list)
    
if __name__ == '__main__':
    train()
