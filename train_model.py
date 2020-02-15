# Training the Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint
import os
import h5py


def mkdir_exists(dir):
    if os.path.exists(dir):
        return
    os.mkdir(dir)


def data_reader():
    with h5py.File(''.join(['dataset-v4.h5']), 'r') as hf:
        X_train = hf['X_train'].value
        y_train = hf['y_train'].value
        X_val = hf['X_val'].value
        y_val = hf['y_val'].value
    return X_train, y_train, X_val, y_val


def train():
    model = build_model.init_model()

    #sgd = SGD(lr=1e-2, decay=1e-4, momentum=0.9, nesterov=True)

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mse']
    )

    X_train, y_train, X_val, y_val = data_reader()

    mkdir_exists("weights")

    # training & validation
    history = model.fit(X_train,
              y_train,
              batch_size=64,
              validation_data=(X_val, y_val),
              epochs=1000,
              verbose=2,
              callbacks=[
                  CSVLogger(
                      'logs.csv',
                      append=True
                  ),
                  ModelCheckpoint(
                      'weights/model-ffn.hdf5',
                      monitor='mean_squared_error',
                      verbose=2,
                      mode='min'
                  )
              ]
              )
    model.summary()
    return history
