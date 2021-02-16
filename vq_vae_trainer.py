import os
import tensorflow as tf
from scipy.io.wavfile import read
from datetime import datetime
import numpy as np


DEFAULT_CONFIGS = {
    'model_path': None,
    'learning_rate': 0.0001,
    'num_epochs': 100,
    'print_every': 100
}

class VQ_VAE_Trainer():

    def __init__(self, model, configs=DEFAULT_CONFIGS):
        self.model = model
        self.configs = configs
        self.reconstr_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.tensorboard_logdir = os.path.join(
            configs['model_path'],
            'tensorboard',
            'run'+datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.file_writer = tf.summary.create_file_writer(
            os.path.join(self.tensorboard_logdir)
        )
        self.file_writer.set_as_default()
        self.ckpt = tf.train.Checkpoint(
            step = tf.Variable(1),
            optimizer = self.optimizer,
            net = model
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, 
            os.path.join(configs['model_path'], 'ckpt'),
            max_to_keep = 3
        )
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")


    def loss_fn(self, data, outputs, z_q, z_e):
        reconstr_loss = self.reconstr_loss(data, outputs)
        vq_loss = tf.reduce_mean(tf.norm(tf.stop_gradient(z_e) - z_q, axis=-1) ** 2)
        commit_loss = tf.reduce_mean(tf.norm(z_e - tf.stop_gradient(z_q), axis=-1) ** 2)
        return reconstr_loss + vq_loss + commit_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            outputs, z_q, z_e = self.model(data)
            loss = self.loss_fn(data, outputs, z_q, z_e)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss, outputs

    def train(self, dataset):
        for i in range(self.configs['num_epochs']):
            for i, data in enumerate(dataset):
                data = mu_law_encode(data, 256, True, False)
                loss, outputs = self.train_step(data)
                print(loss)

                #if i % self.configs['print_every']:
                    ## --------------------------
                    ## TODO:
                    ## --------------------------
                    ## Some visualization
                    ## Convert to audio in juppy
                    ## Do some codebook viz
                    #print(outputs)


def mu_law_encode(x, quantization_channels=256, to_int=False, one_hot=False):
    mu = tf.cast(quantization_channels - 1, tf.float32)
    x = tf.clip_by_value(tf.cast(x, tf.float32), -1., 1.)
    y = tf.sign(x) * tf.math.log1p(mu * tf.abs(x)) / tf.math.log1p(mu)
    if to_int or one_hot:
        # [-1, 1](float) -> (0, mu)(int); + 0.5 since tf.cast does flooring
        y = tf.cast((y + 1) / 2 * mu + 0.5, tf.int32)
        if one_hot:
            y = tf.one_hot(y, depth=quantization_channels, dtype=tf.float32)
            y = tf.squeeze(y, axis=-2)
    print(tf.shape(y))
    return y


def mu_law_decode(y, quantization_channels=256):
    mu = tf.cast(quantization_channels - 1, tf.float32)
    # (0, mu) -> (-1, 1)
    y = (2 * tf.cast(y, tf.float32) / mu) - 1
    x = tf.sign(y) * ((1 + mu) ** tf.abs(y) - 1) / mu
    return x


def mu_law_decode_np(y, quantization_channels=256):
    mu = np.asarray(quantization_channels - 1, dtype=np.float32)
    # (0, mu) -> (-1, 1)
    y = (2 * np.asarray(y, dtype=np.float32) / mu) - 1
    x = np.sign(y) * ((1 + mu) ** abs(y) - 1) / mu
    return x