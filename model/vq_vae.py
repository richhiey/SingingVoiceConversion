import tensorflow as tf
from model.wavenet import WaveNet
from model.conv_encoder import ConvEncoder, ConvDecoder

DEFAULT_CONFIGS = {
	'vq_size': 64,
	'latent_dim': 32
}


class VectorQuantizer(tf.keras.layers.Layer):  
    def __init__(self, k, d, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.k = k
        self.d = d
    
    def build(self, input_shape):
        rand_init = tf.keras.initializers.VarianceScaling(distribution="uniform")
        self.codebook = self.add_weight(
        	shape=(self.k, self.d), initializer=rand_init, trainable=True, name='Codebook'
        )
        
    def call(self, inputs):
        # Map z_e of shape (b, w,, h, d) to indices in the codebook
        lookup_ = tf.reshape(self.codebook, shape=(1, 1, self.k, self.d))
        z_e = tf.expand_dims(inputs, -2)
        dist = tf.norm(z_e - lookup_, axis=-1)
        k_index = tf.argmin(dist, axis=-1)
        return k_index
    
    def sample(self, k_index):
        # Map indices array of shape (b, w, h) to actual codebook z_q
        lookup_ = tf.reshape(self.codebook, shape=(1, 1, self.k, self.d))
        k_index_one_hot = tf.one_hot(k_index, self.k)
        z_q = lookup_ * k_index_one_hot[..., None]
        z_q = tf.reduce_sum(z_q, axis=-2)
        return z_q


class VQ_VAE(tf.keras.Model):

	def __init__(self, configs=DEFAULT_CONFIGS):
		super(VQ_VAE, self).__init__()
		self.model = self.create_model(configs)


	def create_model(self, configs):
		self.conv_encoder = ConvEncoder()
		self.wavenet_decoder = WaveNet()
		self.conv_decoder = ConvDecoder()
		self.vector_quantizer = VectorQuantizer(
			configs['vq_size'],
			configs['latent_dim']
		)
		# Input audio : sr = 16KHz
		# - (batch, sr, 1)
		input_audio = tf.keras.layers.Input(shape=(None, 1))
		# Conv Encoder
		# - (batch, m, 1)
		encoded = self.conv_encoder(input_audio)
		# Pre-VQ step
		# - (batch, m, latent size)
		z_e = tf.keras.layers.Conv1D(configs['latent_dim'], 1, name='Pre-VQ-conv')(encoded)
		# Vector quantization
		# - (batch, m, latent size)
		z_q = self.vector_quantizer(encoded)
    	straight_through = tf.keras.layers.Lambda(
    		lambda x : x[1] + tf.stop_gradient(x[0] - x[1]),
    		name="straight_through_estimator"
    	)
    	vq = straight_through([z_q, z_e])
		# Conv Decoder
		# - (batch, sr, 1)

		decoded = self.conv_decoder(vq)
		# Wavenet Decoder
		# - (batch, sr, 1)
		output = self.wavenet_decoder(decoded)

		model = tf.keras.Model(input_audio, output)
		model.summary()

		return model


	def call(self, inputs):
		return self.model(inputs)

