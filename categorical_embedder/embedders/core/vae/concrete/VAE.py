from keras.layers import Layer
from keras import backend as K

from categorical_embedder.embedders.core.vae.concrete.VAEEncoder import VAEEncoder
from categorical_embedder.embedders.core.vae.concrete.VAEDecoder import VAEDecoder
from categorical_embedder.embedders.core.aux.loss_factory import get_loss_function


class VAE(Layer):
    def __init__(self, seq_length, feature_dim, hidden_length, encoder_latent_info, encoder_layer_info, decoder_layer_info, inner_loss_info, **kwargs):
        super().__init__(**kwargs)
        self._seq_length = seq_length
        self._feature_dim = feature_dim
        self._hidden_length = hidden_length
        self._encoder_latent_info = encoder_latent_info
        self._encoder_layer_info = encoder_layer_info
        self._decoder_layer_info = decoder_layer_info
        self._inner_loss_info = inner_loss_info
        self._reconstruction_loss_fn = get_loss_function(self._inner_loss_info["reconstruction_loss"])

        self._encoder = VAEEncoder(self._encoder_layer_info, self._hidden_length, self._encoder_latent_info.get("continuous_latent_length", None), self._encoder_latent_info.get("discrete_latent_length", None), self._encoder_latent_info.get("temperature", 0.1), self._inner_loss_info["z_loss_weight"], self._inner_loss_info["c_loss_weight"])
        self._decoder = VAEDecoder(self._decoder_layer_info, self._hidden_length, (self._seq_length, self._feature_dim))

    def call(self, inputs, **kwargs):
        encoded = self._encoder(inputs)
        decoder_output = self._decoder(encoded)

        # As we are trying to get the input back, y_true = inputs
        # Assuming 3D loss, summ over sequence and mean over batches.
        self.add_loss(self._inner_loss_info["r_loss_weight"] * K.mean(K.sum(self._reconstruction_loss_fn(K.cast_to_floatx(inputs), decoder_output), axis=-1)))

        return decoder_output, encoded

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "seq_length": self._seq_length,
            "feature_dim": self._feature_dim,
            "hidden_length": self._hidden_length,
            "encoder_latent_info": self._encoder_latent_info,
            "encoder_layer_info": self._encoder_layer_info,
            "decoder_layer_info": self._decoder_layer_info,
            "inner_loss_info": self._inner_loss_info
        }
        base_config = super().get_config()
        config.update(base_config)

        return config

    def get_encoder(self):
        return self._encoder
