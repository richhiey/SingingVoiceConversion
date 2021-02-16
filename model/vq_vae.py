import tensorflow as tf
from model.wavenet import WaveNet
from model.conv_encoder import ConvEncoder, ConvDecoder

DEFAULT_CONFIGS = {
    'vq_size': 512,
    'latent_dim': 128
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
        input_audio = tf.keras.layers.Input(shape=(None, 1))
        # - (batch, sr, 1)
        # Conv Encoder
        encoded = self.conv_encoder(input_audio)
        # - (batch, m, 1)
        # Pre-VQ step
        z_e = tf.keras.layers.Conv1D(configs['latent_dim'], 1, name='pre-vq-conv')(encoded)
        # - (batch, m, latent size)
        # Vector quantization
        codebook_idx = self.vector_quantizer(z_e)
        # - (batch, m)
        sampling_layer = tf.keras.layers.Lambda(
            lambda x: self.vector_quantizer.sample(x),
            name="sample_from_codebook"
        )
        # Replace indicies with codebook vectors
        z_q = sampling_layer(codebook_idx)
        # - (batch, m, latent_size)
        # Stop gradient for vq step
        straight_through = tf.keras.layers.Lambda(
            lambda x : x[1] + tf.stop_gradient(x[0] - x[1]),
            name="straight_through_estimator"
        )
        vq = straight_through([z_q, z_e])
        # Conv Decoder
        decoded = self.conv_decoder(vq)
        # - (batch, sr, 1)
        # Wavenet Decoder
        output = self.wavenet_decoder(decoded)
        # - (batch, sr, 1)

        model = tf.keras.Model(input_audio, [output, z_q, z_e])
        model.summary()

        return model


    def call(self, inputs):
        return self.model(inputs)

