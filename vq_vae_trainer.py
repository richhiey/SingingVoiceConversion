import os
import tensorflow as tf

from datetime import datetime

DEFAULT_CONFIGS = {
    'model_path': '/home/richhiey/Desktop/workspace/projects/virtual_musicians',
    'learning_rate': 0.0001,
    'num_epochs': 100,
    'print_every': 100
}

class VQ_VAE_Trainer():

    def __init__(self, model, configs=DEFAULT_CONFIGS):
        self.model = model
        self.configs = configs
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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


    def train_step(self, data):
        with tf.GradientTape() as tape:
            outputs = self.model(data)
            loss = self.loss_fn(outputs, data)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss, outputs

    def train(self, dataset):
        for i in range(self.configs['num_epochs']):
            for i, data in enumerate(dataset):
                loss, outputs = self.train_step(data)
                print(loss)

                if i % self.configs['print_every']:
                    ## --------------------------
                    ## TODO:
                    ## --------------------------
                    ## Some visualization
                    ## Convert to audio in juppy
                    ## Do some codebook viz
                    print(outputs)

