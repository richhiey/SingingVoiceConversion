import tensorflow as tf

DEFAULT_CONFIGS = {
	'num_layers': 6,
	'num_filters': 256,
	'kernel_size': 4,
	'strides': 2
}

class ConvDecoder(tf.keras.Model):

	def __init__(self, configs=DEFAULT_CONFIGS):
		super(ConvDecoder, self).__init__()
		self.model = self.create_model(configs)

	def create_model(self, configs):
		encoded = tf.keras.layers.Input(shape=(None, 1))
		x = encoded

		for i in range(configs['num_layers']):
			x = tf.keras.layers.Conv1DTranspose(
                filters=configs['num_filters'],
                kernel_size=configs['kernel_size'],
                strides=configs['strides'],
                padding='same',
                activation='relu'
			)(x)
		
		out = tf.keras.layers.Conv1D(
			filters=1,
			kernel_size=1,
			activation='relu'
		)(x)

		model = tf.keras.Model(encoded, out, name='Convolutional-Decoder')
		model.summary()

		return model

	def call(self, inputs):
		return self.model(inputs)


class ConvEncoder(tf.keras.Model):

	def __init__(self, configs=DEFAULT_CONFIGS):
		super(ConvEncoder, self).__init__()
		self.model = self.create_model(configs)

	def create_model(self, configs):
		input_audio = tf.keras.layers.Input(shape=(16000, 1))
		x = input_audio
		for i in range(configs['num_layers']):
			x = tf.keras.layers.Conv1D(
                filters=configs['num_filters'],
                kernel_size=4,
                strides=2,
                padding='same',
                activation='relu'
			)(x)

		out = tf.keras.layers.Conv1D(
			filters=1,
			kernel_size=1,
			activation='relu'
		)(x)
		model = tf.keras.Model(input_audio, out, name='Convolutional-Encoder')
		model.summary()

		return model

	def call(self, inputs):
		return self.model(inputs)
