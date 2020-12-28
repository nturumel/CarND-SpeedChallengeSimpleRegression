from globals import INPUT_SHAPE, TEST_OUTPUT, SAVE_ARRAY_FILE, MODEL_PATH

from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import numpy as np

def model():
    model = Sequential()
    model.add(Input(shape(INPUT_SHAPE)))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1), activation='relu' )
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train(train_input = SAVE_ARRAY_FILE, test_output = TEST_OUTPUT):
    # split into input (X) and output (Y) variables
    X = np.load(train_input)
    file1 = open(TEST_OUTPUT, 'r') 
    Y = file1.readline()

    # train the model
    estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=250, verbose=0)
    kfold = KFold(n_splits=10)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1,
          save_best_only=True, mode='auto', period=1)
    results = cross_val_score(estimator, X, Y, cv=kfold, fit_params={'callbacks': [checkpoint(), TensorBoard()]})
    print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

if __name__ == '__main__':
    train()
