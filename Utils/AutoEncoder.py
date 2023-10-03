import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras import layers, losses, Sequential
from keras.layers import Flatten, Dense, Dropout, Activation, Reshape, LeakyReLU as LR
from keras.models import Model
import numpy as np


class Autoencoder(Model):
    def __init__(self, latent_dim, row_num, col_num):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='sigmoid')
            # relu, elu, sigmoid, tanh, tf.keras.activations.tanh, tf.keras.layers.LeakyReLU(alpha=0.3)
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(row_num * col_num, activation='sigmoid'),
            layers.Reshape((row_num, col_num))
        ])

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def encode_table(table_data, eval_data, latent_dim, epoch_no):
    table_data_array = np.array(table_data)
    eval_data_array = np.array(eval_data)
    print(table_data_array.shape)
    print(eval_data_array.shape)
    autoencoder = Autoencoder(latent_dim, table_data_array.shape[1], table_data_array.shape[2])
    if eval_data:
        early_stopping = EarlyStopping(monitor='val_loss', patience=25, mode='min')
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])
        autoencoder.fit(table_data_array, table_data_array,
                        validation_data=(eval_data_array, eval_data_array),
                        epochs=epoch_no,
                        callbacks=[early_stopping])
    else:
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])
        autoencoder.fit(table_data_array, table_data_array,
                        validation_data=(eval_data_array, eval_data_array),
                        epochs=epoch_no)
    return autoencoder
