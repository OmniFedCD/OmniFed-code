import time, os

import numpy as np

from utils import get_data
from data import SlidingWindowDataset, SlidingWindowDataLoader

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, Model

class Encoder(Model):
    def __init__(self, input_dims: int, z_dims: int, nn_size = None):
        super().__init__()
        if not nn_size:
            nn_size = (input_dims // 2, input_dims // 4)

        self.model = Sequential()
        for cur_size in nn_size:
            self.model.add(layers.Dense(cur_size, activation="relu",))

        self.model.add(layers.Dense(z_dims, activation="relu",))

    def call(self, x):
        z = self.model(x)
        return z

class Decoder(Model):
    def __init__(self, input_dims: int, z_dims: int, nn_size = None):
        super().__init__()

        if not nn_size:
            nn_size = (input_dims // 4, input_dims // 2)

        self.model =  Sequential()
        for cur_size in nn_size:
            self.model.add(layers.Dense(cur_size, activation="relu"))
        
        self.model.add(layers.Dense(input_dims, activation="tanh"))

    def call(self, z):
        w = self.model(z)
        return w

class USAD():

    def __init__(self, x_dims: int, max_epochs: int = 250, batch_size: int = 128,
                 encoder_nn_size = None, decoder_nn_size = None,
                 z_dims: int = 38, window_size: int = 10, message = None, train_num = None):

        self._x_dims = x_dims
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._encoder_nn_size = encoder_nn_size
        self._decoder_nn_size = decoder_nn_size
        self._z_dims = z_dims
        self._window_size = window_size
        self._input_dims = x_dims * window_size
        self._message = message
        self._train_num = train_num

        self._shared_encoder = Encoder(input_dims=self._input_dims, z_dims=self._z_dims)
        self._decoder_G = Decoder(z_dims=self._z_dims, input_dims=self._input_dims)
        self._decoder_D = Decoder(z_dims=self._z_dims, input_dims=self._input_dims)


    def fit(self, values, valid_portion=0.2, verbose=False):
        n = int(len(values) * valid_portion)
        if n == 0:
            train_values, valid_values = values[:], values[-1:]
        else:
            train_values, valid_values = values[:-n], values[-n:]

        train_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(train_values, self._window_size)._strided_values,
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=True
        )

        valid_sliding_window =  SlidingWindowDataLoader(
            SlidingWindowDataset(valid_values, self._window_size)._strided_values,
            batch_size=self._batch_size
        )

        mse = tf.keras.losses.MeanSquaredError()
        optimizer_G = tf.keras.optimizers.Adam(lr=0.001)
        optimizer_D = tf.keras.optimizers.Adam(lr=0.001)

        train_time = 0
        valid_time = 0

        for epoch in range(1, self._max_epochs + 1):
            
            train_start = time.time()
            for step in range(train_sliding_window.total):
                
                with tf.GradientTape(persistent=True) as tape:
                    x_batch_train = train_sliding_window.get_item(step)
                    w = tf.reshape(x_batch_train,(-1, self._input_dims))

                    z = self._shared_encoder(w)
                    w_G = self._decoder_G(z)
                    w_D = self._decoder_D(z)
                    w_G_D = self._decoder_D(self._shared_encoder(w_G))

                    loss_G = (1 / epoch) * mse(w_G, w) + (1 - 1 / epoch) * mse(w_G_D, w)
                    loss_D = (1 / epoch) * mse(w_D, w) - (1 - 1 / epoch) * mse(w_G_D, w)

                grad_ae_G = tape.gradient(loss_G, self._shared_encoder.trainable_variables + self._decoder_G.trainable_variables + self._decoder_D.trainable_variables)
                grad_ae_D = tape.gradient(loss_D, self._shared_encoder.trainable_variables + self._decoder_G.trainable_variables + self._decoder_D.trainable_variables)

                grad_E, grad_G, grad_D = [], [], []
                    
                for i in range(len(grad_ae_G)):
                    if i < len(grad_ae_G)//3:
                        grad_E.append(grad_ae_G[i] + grad_ae_D[i])
                    elif i < 2*len(grad_ae_G)//3:
                        grad_G.append(grad_ae_G[i] + grad_ae_D[i])
                    else:
                        grad_D.append(grad_ae_G[i] + grad_ae_D[i])

                optimizer_G.apply_gradients(zip(grad_E + grad_G, self._shared_encoder.trainable_variables + self._decoder_G.trainable_variables))
                optimizer_D.apply_gradients(zip(grad_E + grad_D, self._shared_encoder.trainable_variables + self._decoder_D.trainable_variables))
                del tape

            train_time += time.time() - train_start

            val_losses_G = []
            val_losses_D = []
            valid_start = time.time()

            for step in range(valid_sliding_window._total):
                x_batch_val = valid_sliding_window.get_item(step)

                w = tf.reshape(x_batch_val,(-1, self._input_dims))
                z = self._shared_encoder(w)
                w_G = self._decoder_G(z)
                w_D = self._decoder_D(z)
                w_G_D = self._decoder_D(self._shared_encoder(w_G))

                val_loss_G = 1 / epoch * tf.reduce_mean((w - w_G) ** 2) + (1 - 1 / epoch) * tf.reduce_mean(
                    (w - w_G_D) ** 2)
                val_loss_D = 1 / epoch * tf.reduce_mean((w - w_D) ** 2) - (1 - 1 / epoch) * tf.reduce_mean(
                    (w - w_G_D) ** 2)

                val_losses_G.append(val_loss_G.numpy())
                val_losses_D.append(val_loss_D.numpy())
            
            valid_time += time.time() - valid_start
            
            val_G_loss = np.mean(val_losses_G)
            val_D_loss = np.mean(val_losses_D)

            if verbose:
                print(f'{self._message} epoch {epoch} val_G_loss: {val_G_loss}, val_D_loss: {val_D_loss}')
        if verbose:
            print()

        return {
            'train_time': train_time,
            'valid_time': valid_time
        }

    def predict(self, values, alpha=1, beta=0, on_dim=False):
        collect_scores = []
        test_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(values, self._window_size)._strided_values,
            batch_size=self._batch_size
        )
        
        for step in range(test_sliding_window._total):
            w = test_sliding_window.get_item(step)
            w = tf.reshape(w, (-1, self._input_dims))

            z = self._shared_encoder(w)
            w_G = self._decoder_G(z)
            w_D = self._decoder_D(z)
            w_G_D = self._decoder_D(self._shared_encoder(w_G))

            batch_scores = alpha *((w - w_G) ** 2) + beta * ((w - w_G_D) ** 2)
            batch_scores = tf.reshape(batch_scores, (-1, self._window_size, self._x_dims))

            if not on_dim:
                batch_scores = np.sum(batch_scores, axis=2)
            collect_scores.extend(batch_scores[:, -1])

        return np.array(collect_scores)

    def reconstruct(self, values):
        collect_G = []
        collect_G_D = []
        collect_D = []
        test_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(values, self._window_size)._strided_values,
            batch_size=self._batch_size
        )

        n = 0
        for step in range(test_sliding_window._total):
            w = test_sliding_window.get_item(step)
            w = tf.reshape(w,(-1, self._input_dims))

            z = self._shared_encoder(w)
            w_G = self._decoder_G(z)
            w_D = self._decoder_D(z)
            w_G_D = self._decoder_D(self._shared_encoder(w_G))

            batch_G = tf.reshape(w_G, (-1, self._window_size, self._x_dims))
            batch_G_D = tf.reshape(w_G_D, (-1, self._window_size, self._x_dims))
            batch_D =  tf.reshape(w_D, (-1, self._window_size, self._x_dims))
            
            collect_G.extend(batch_G[:, -1])
            collect_G_D.extend(batch_G_D[:, -1])
            collect_D.extend(batch_D[:, -1])
        
        return np.array(collect_G), np.array(collect_G_D), np.array(collect_D)

    def save(self, savedir):
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        self._shared_encoder.save_weights(os.path.join(savedir, 'shared_encoder'))
        self._decoder_G.save_weights(os.path.join(savedir, 'decoder_G'))
        self._decoder_D.save_weights(os.path.join(savedir, 'decoder_D'))


    def restore(self, restoredir):
        self._shared_encoder.load_weights(os.path.join(restoredir, 'shared_encoder')).expect_partial()
        self._decoder_G.load_weights(os.path.join(restoredir, 'decoder_G')).expect_partial()
        self._decoder_D.load_weights(os.path.join(restoredir, 'decoder_D')).expect_partial()


    def build(self):
        w = tf.reshape(tf.ones((self._input_dims)), (-1, self._input_dims))

        z = self._shared_encoder(w)
        w_G = self._decoder_G(z)
        w_D = self._decoder_D(z)
        w_G_D = self._decoder_D(self._shared_encoder(w_G))