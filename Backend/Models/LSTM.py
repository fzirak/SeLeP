import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, RepeatVector, Reshape, Flatten, Concatenate, Embedding, Conv2D, \
    MaxPooling2D, TimeDistributed
from keras.models import Model, model_from_json, Sequential



from typing import List
import pandas as pd
import numpy as np
import math
import csv
csv.field_size_limit(100 * 1024 * 1024) 
import _pickle as cPickle
import time
from tqdm import tqdm
import pprint as pp


class ActionTimestamp:
    def __init__(self, name, begin=0.0, end=0.0):
        self.name = name
        self.begin = begin
        self.end = end

    def get_duration(self):
        return self.end - self.begin

    def __str__(self) -> str:
        return f'{self.name}: {self.get_duration()}'

    def __repr__(self) -> str:
        return f'{self.name}: {self.get_duration()}'


def create_binary_lstm_model(num_partitions, look_back, rows, cols):
    """
    this function returns an encoder_decoder LSTM model. input shape is (None, lookback, rows*cols).
    The output shape is (None, num_partitions)
    """
    data_shape = rows * cols
    encoder_input = Input(shape=(rows, cols))
    enc_in = Reshape(target_shape=(rows, cols, 1))(encoder_input)  # Reshape the input for Conv2D
    enc_h1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(enc_in)
    enc_h1 = MaxPooling2D(pool_size=(2, 2))(enc_h1)
    enc_h1 = Flatten()(encoder_input)
    enc_out = Dense(units=128, activation='relu')(enc_h1)
    conv_encoder = Model(inputs=encoder_input, outputs=enc_out)

    # Model input
    model_input = Input(shape=(look_back, data_shape))
    reshape_input = Reshape((look_back, rows, cols))(model_input)

    encoded_matrices = []
    for i in range(look_back):
        enc_out = conv_encoder(reshape_input[:, i, :, :])  # Encode each matrix separately
        encoded_matrices.append(enc_out)

    # LSTM model
    lstm_input = tf.stack(encoded_matrices, axis=1)

    x, last_h, last_c = LSTM(64, return_state=True, name='enc_lstm_2')(lstm_input)
    x = RepeatVector(1, name='repeat_1')(x)
    x = LSTM(64, return_sequences=True, name='dec_lstm_1')(x, initial_state=[last_h, last_c])
    x = Dense(units=num_partitions, activation='sigmoid', name='dense_2')(x)
    output = Flatten()(x)

    # Create the model
    model = Model(inputs=model_input, outputs=output)
    return model


def create_vanilla_lstm_model(look_back, rows, cols):
    data_shape = rows * cols
    model = Sequential()
    model.add(LSTM(64, input_shape=(look_back, data_shape), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(units=data_shape))
    model.add(Reshape((1, data_shape)))
    return model


def store_model(model, model_name, base_model_file_dir):
    model_json = model.to_json()
    with open(f"{base_model_file_dir}{model_name}.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(f"{base_model_file_dir}{model_name}.h5")
    print(f"Saved {model_name} to disk")


def load_model(model_name, base_model_file_dir):
    json_file = open(f'{base_model_file_dir}{model_name}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f"{base_model_file_dir}{model_name}.h5")
    print(f"Loaded {model_name} from disk")
    return loaded_model