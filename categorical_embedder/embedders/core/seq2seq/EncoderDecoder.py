import copy
from keras.layers import Layer, TimeDistributed, Dense

from categorical_embedder.embedders.core.seq2seq.StackedRecurrentEncoder import StackedRecurrentEncoder
from categorical_embedder.embedders.core.seq2seq.StackedRecurrentDecoder import StackedRecurrentDecoder


class EncoderDecoder(Layer):
    def __init__(self, feature_dimension, embedding_length, output_layer_activation, encoder_info, decoder_info, **kwargs):
        super().__init__(**kwargs)
        self._feature_dimension = feature_dimension
        self._embedding_length = embedding_length
        self._output_layer_activation = output_layer_activation

        self._encoder_info = copy.deepcopy(encoder_info)
        self._encoder_info["units"] = self._embedding_length

        self._decoder_info = copy.deepcopy(decoder_info)
        self._decoder_info["units"] = self._embedding_length

        self._encoder = StackedRecurrentEncoder(**self._encoder_info)
        self._decoder = StackedRecurrentDecoder(**self._decoder_info)

        self._output_layer = TimeDistributed(Dense(self._feature_dimension, activation=self._output_layer_activation))

    def call(self, inputs, **kwargs):
        # Be cautious when using normalize with LSTM.
        # The effect of normalizing cell state along with hidden state is not straightforward to see.

        # inputs: [encoder_input, decoder_input]

        encoded = self._encoder(inputs[0])
        decoder_output = self._decoder(inputs[1], initial_state=encoded)
        output = self._output_layer(decoder_output)
        return output, encoded[0]

    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1] + (self._feature_dimension,), self._encoder.compute_output_shape(input_shape[0])

    def get_config(self):
        config = {
            "feature_dimension": self._feature_dimension,
            "embedding_length": self._embedding_length,
            "output_layer_activation": self._output_layer_activation,
            "encoder_info": self._encoder_info,
            "decoder_info": self._decoder_info
        }
        base_config = super().get_config()
        config.update(base_config)

        return config

    def get_encoder(self):
        return self._encoder
