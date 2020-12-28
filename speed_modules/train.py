from globals import INPUT_SHAPE, TRAIN_OUTPUT, SAVE_ARRAY_FILE, MODEL_PATH, LOG_DIR

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Activation
from keras import activations
from keras.callbacks import ModelCheckpoint, TensorBoard, History 
import keras.metrics
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import numpy as np
import csv

def model():
    model = Sequential()
    model.add(Input(shape = (INPUT_SHAPE)))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'MeanSquaredError'])
    return model

def train(train_input = SAVE_ARRAY_FILE, train_output = TRAIN_OUTPUT):
    # split into input (X) and output (Y) variables
    X = np.load(train_input)
    #FIXME: Simplify
    Y = []
    with open(train_output) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for item in row:
                Y.append(float(item))
    Y = np.array(Y[:-1])

    # train the model
    estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=250, verbose=0)
    kfold = KFold(n_splits=10)
    checkpoint = ModelCheckpoint(MODEL_PATH, verbose=1, monitor='loss')
    tensorboard = TensorBoard(log_dir=LOG_DIR)
    history = History()
    callbacks_list = [checkpoint, tensorboard, history]
    results = cross_val_score(estimator, X, Y, cv=kfold, fit_params={'callbacks': callbacks_list})
    print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()), end="\n")

if __name__ == '__main__':
    train()
