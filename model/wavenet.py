import tensorflow as tf

DEFAULT_CONFIGS = {}

class WaveNet(tf.keras.Model):
    def __init__(self, configs):
        super(WaveNetPedal, self).__init__()
        self.model = self.create_model(configs)


    def create_model(self, configs):
        skips = []
        input_audio = tf.keras.layers.Input(shape=(None, 1))
        x = input_audio

        for d in configs['dilation_rates']:
            # preprocessing eqv. in time-distributed dense
            x = tf.keras.layers.Conv1D(
                16, 1, padding='same', activation='relu'
            )(x)
            # Filter
            x_f = tf.keras.layers.Conv1D(
                filters=configs['num_filters'],
                kernel_size=configs['kernel_size'],
                padding='causal',
                dilation_rate=d
            )(x)
            # Gate
            x_g = tf.keras.layers.Conv1D(
                filters=configs['num_filters'],
                kernel_size=configs['kernel_size'],
                padding='causal',
                dilation_rate=d
            )(x)
            x_f = tf.keras.layers.Activation('tanh')(x_f)
            x_g = tf.keras.layers.Activation('sigmoid')(x_g)
            z = tf.keras.layers.Multiply()([x_f, x_g])
            # post-processing eqv. in time-distributed dense
            z = tf.keras.layers.Conv1D(
                16, 1, padding='same', activation='relu'                
            )(z)
            x = tf.keras.layers.Add()([x, z])
            skips.append(z)

        out = tf.keras.layers.Add()(skips)
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.Conv1D(128, 1, padding='same')(out)
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.Dropout(.2)(out)
        out = tf.keras.layers.Conv1D(1, 1, padding='same')(out)
        model = tf.keras.Model(input_audio, out)
        model.summary()
        return model

    def call(self, inputs):
        return self.model(inputs)
