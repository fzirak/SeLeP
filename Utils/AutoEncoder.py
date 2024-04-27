import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras import layers, losses, Sequential
from keras.layers import Flatten, Dense, Dropout, Activation, Reshape, LeakyReLU as LR
from keras.models import Model
import numpy as np


class Autoencoder(Model):
    def __init__(self, latent_dim, row_num, col_num, encoder_option=0):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        if encoder_option == 0: #SLP
            self.encoder = tf.keras.Sequential([
                layers.Flatten(),
                layers.Dense(latent_dim, activation='sigmoid')
            ])
            self.decoder = tf.keras.Sequential([
                layers.Dense(row_num * col_num, activation='sigmoid'),
                layers.Reshape((row_num, col_num))
            ])
        elif encoder_option == 1: #MLP
            self.encoder = tf.keras.Sequential([
                layers.Flatten(),
                layers.Dense(latent_dim*2, activation='relu'),
                layers.Dense(latent_dim, activation='sigmoid')
                # relu, elu, sigmoid, tanh, tf.keras.activations.tanh, tf.keras.layers.LeakyReLU(alpha=0.3)
            ])
            self.decoder = tf.keras.Sequential([
                layers.Dense(row_num * col_num / 2, activation='relu'),
                layers.Dense(row_num * col_num, activation='sigmoid'),
                layers.Reshape((row_num, col_num))
            ])
        elif encoder_option == 2: #Conv
            self.encoder = tf.keras.Sequential([
            layers.Reshape((row_num, col_num, 1), input_shape=(row_num, col_num), name='repeat1'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='sigmoid')
            ])
            self.decoder = tf.keras.Sequential([
                layers.Reshape((1, 1, latent_dim), input_shape=(latent_dim,), name='repeat2'),
                layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
                layers.UpSampling2D((2, 2)),
                layers.Conv2DTranspose(8, (3, 3), activation='sigmoid', padding='same'),
                layers.UpSampling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(row_num*col_num, activation='sigmoid'),
                layers.Reshape((row_num, col_num), name='repeat3')
            ])
        elif encoder_option == 3:
            self.encoder = tf.keras.Sequential([
                layers.Flatten(),
                layers.Dense(latent_dim, activation='hard_sigmoid')
            ])
            self.decoder = tf.keras.Sequential([
                layers.Dense(row_num * col_num, activation='hard_sigmoid'),
                layers.Reshape((row_num, col_num))
            ])

        

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def encode_table(table_data, eval_data, latent_dim, epoch_no, encoder_option):
    # print(np.array(table_data).shape)
    table_data_array = np.array(table_data)
    eval_data_array = np.array(eval_data)
    print('\t train data shape: ', table_data_array.shape)
    print('\t validation data shape: ', eval_data_array.shape)
    autoencoder = Autoencoder(latent_dim, table_data_array.shape[1], table_data_array.shape[2], encoder_option)
    if eval_data:
        early_stopping = EarlyStopping(monitor='val_loss', patience=25, mode='min')
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy', 'mse'])
        autoencoder.fit(table_data_array, table_data_array,
                        validation_data=(eval_data_array, eval_data_array),
                        epochs=epoch_no,
                        # shuffle=True,
                        callbacks=[early_stopping])
    else:
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy', 'mse'])
        autoencoder.fit(table_data_array, table_data_array,
                        validation_data=(eval_data_array, eval_data_array),
                        epochs=epoch_no)
    return autoencoder
