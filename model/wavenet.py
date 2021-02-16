# Implementation of original Wavenet paper
# - https://arxiv.org/pdf/1609.03499.pdf

import tensorflow as tf

DEFAULT_CONFIGS = {
    "filter_width": 4,
    "sampling_rate": 16000,
    "dilations": [
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
      #  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
      # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
      # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
      # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
    ],
    "residual_channels": 32,
    "dilation_channels": 32,
    "quantization_channels": 256,
    "skip_channels": 128,
    "initial_filter_width": 32,
    "initial_channels": 32
}


class WaveNet(tf.keras.Model):
    def __init__(self, configs=DEFAULT_CONFIGS):
        super(WaveNet, self).__init__()
        self.model = self.create_model(configs)


    def create_model(self, configs):
        skips = []
        input_audio = tf.keras.layers.Input(shape=(configs['sampling_rate'], 1))

        x = tf.keras.layers.Conv1D(
            filters=configs['initial_channels'], 
            kernel_size=configs['initial_filter_width'],
            padding='causal',
            activation='relu',
            name='Initial-Causal-Convolution'
        )(input_audio)

        for i, d in enumerate(configs['dilations']):
            # preprocessing eqv. in time-distributed dense
            # Filter
            x_f = tf.keras.layers.Conv1D(
                filters=configs['dilation_channels'],
                kernel_size=configs['filter_width'],
                padding='causal',
                dilation_rate=d,
                name='Layer'+str(i)+'-Conv-Filter'
            )(x)
            # Gate
            x_g = tf.keras.layers.Conv1D(
                filters=configs['dilation_channels'],
                kernel_size=configs['filter_width'],
                padding='causal',
                dilation_rate=d,
                name='Layer'+str(i)+'-Conv-Gate'
            )(x)
            x_f = tf.keras.layers.Activation('tanh')(x_f)
            x_g = tf.keras.layers.Activation('sigmoid')(x_g)
            z = tf.keras.layers.Multiply()([x_f, x_g])
            z = tf.keras.layers.Conv1D(
                filters=configs['residual_channels'],
                kernel_size=1,
                padding='same',
                activation='relu',
                name='Layer'+str(i)+'-Conv-Residual'              
            )(z)
            x = tf.keras.layers.Add()([x, z])
            skips.append(z)

        out = tf.keras.layers.Add()(skips)
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.Conv1D(
            filters=configs['skip_channels'],
            kernel_size=1,
            padding='same',
            activation='relu',
            name='Conv-Skip-Channels'
        )(out)
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.Dropout(.2)(out)
        
        final_layer_name = 'Final-Conv-Layer'
        if configs.get('quantization_channels'):
            out = tf.keras.layers.Conv1D(
                filters=configs['quantization_channels'],
                kernel_size=1,
                padding='same',
                name=final_layer_name
            )(out)
            out = tf.keras.layers.Softmax()(out)
        else:
            out = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=1,
                padding='same',
                name=final_layer_name
            )(out)

        model = tf.keras.Model(input_audio, out, name='WaveNet')
        model.summary()
        return model

    def call(self, inputs):
        return self.model(inputs)
