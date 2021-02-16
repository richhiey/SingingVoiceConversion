import os
import tensorflow as tf
from scipy.io.wavfile import read
from datetime import datetime
import numpy as np
import IPython.display as ipd


DEFAULT_CONFIGS = {
    'model_path': None,
    'learning_rate': 0.0001,
    'num_epochs': 100,
    'print_every': 1000,
    'save_every': 1000,
}

class WaveNet_Trainer():

    def __init__(self, model, configs=DEFAULT_CONFIGS):
        self.model = model
        self.configs = configs
        self.reconstr_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(configs['learning_rate'])
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


    def loss_fn(self, data, outputs):
        reconstr_loss = self.reconstr_loss(data, outputs)
        return reconstr_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            outputs = self.model(data)
            loss = self.loss_fn(data, outputs)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss, outputs

    def train(self, dataset):
        for i in range(self.configs['num_epochs']):
            for i, _data in enumerate(dataset):
                data = mu_law_encode(_data, 256, True, False)
                loss, outputs = self.train_step(data)
                
                self.ckpt.step.assign_add(1)
                curr_step = int(self.ckpt.step)
                tf.summary.scalar('Loss', loss, step=curr_step)
                if curr_step % self.configs['print_every'] == 0:
                    ## --------------------------
                    ## TODO:
                    ## --------------------------
                    ## Some visualization
                    ## Convert to audio in juppy
                    ## Do some codebook viz
                    # print(outputs)
                    print(loss)
                    wav = tf.argmax(outputs, axis=-1)
                    pred = mu_law_decode_np(tf.squeeze(wav[0,...]).numpy())
                    og = mu_law_decode_np(tf.squeeze(data[0,...]).numpy())
                    print('PREDICTED:')
                    ipd.display(ipd.Audio(pred, rate=16000))
                    print('ORIGINAL (Mu-law):')
                    ipd.display(ipd.Audio(og, rate=16000))
                    print('ORIGINAL:')
                    ipd.display(ipd.Audio(tf.squeeze(_data[0,...]).numpy(), rate=16000))
                    
                if curr_step % self.configs['save_every'] == 0:
                    self.ckpt_manager.save()
                    print('Saved checkpoints for step: ' + str(curr_step))


def mu_law_encode(x, quantization_channels=256, to_int=False, one_hot=False):
    mu = quantization_channels - 1
    x = tf.cast(x, dtype=tf.float32) / (2 ** 16 - 1)
    safe_audio_abs = tf.minimum(tf.abs(x), 1.0)
    magnitude = tf.math.log(1. + mu * safe_audio_abs) / tf.math.log(1. + mu)
    signal = tf.sign(x) * magnitude
    return tf.cast((signal + 1) / 2 * mu + 0.5, tf.int32)


def mu_law_decode(y, quantization_channels=256):
    mu = quantization_channels - 1
    signal = 2 * (tf.to_float(y) / mu) - 1
    magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
    return tf.sign(signal) * magnitude


def mu_law_decode_np(y, quantization_channels=256):
    mu = np.asarray(quantization_channels - 1, dtype=np.float32)
    # (0, mu) -> (-1, 1)
    y = (2 * np.asarray(y, dtype=np.float32) / mu) - 1
    x = np.sign(y) * ((1 + mu) ** abs(y) - 1) / mu
    return x