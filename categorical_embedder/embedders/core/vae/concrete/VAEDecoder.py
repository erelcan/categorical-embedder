from keras.layers import Layer, Dense

from categorical_embedder.embedders.core.aux import layer_creator


class VAEDecoder(Layer):
    def __init__(self, layer_info, hidden_length, output_shape, **kwargs):
        super().__init__(**kwargs)

        self._layer_info = layer_info
        self._hidden_length = hidden_length
        self._output_shape = output_shape

        self._reverse_hidden_layer = Dense(self._hidden_length, activation='relu')

        self._custom_layers = []
        for info in self._layer_info:
            self._custom_layers.append(layer_creator.create(info["type"], info["parameters"]))

    def call(self, inputs, **kwargs):
        # Encoding is to be passed as the input.
        output = self._reverse_hidden_layer(inputs)

        for cur_layer in self._custom_layers:
            output = cur_layer(output)

        return output

    def compute_output_shape(self, input_shape):
        # For categoricals, we expect 3D input which is a batch of sequences of one-hot-vectors.
        # The output for that will be 3D which is a batch of sequences of (preferably softmax-ed)logit arrays
        # For time series, we expect an input 3D input which is a batch of sequences of features/values.
        # The number of values ranges from 1 to k representing the number of series.
        # The output is 3D, a batch of sequences of logits. last dimension has a length equal to k.
        # Note that it is better not to apply softmax on the logits. However, if the input is converted to
        # a distribution and the output is to be reverse-mapped; then we may have it.
        return (input_shape[0],) + self._output_shape

    def get_config(self):
        config = {
            "layer_info": self._layer_info,
            "hidden_length": self._hidden_length
        }
        base_config = super().get_config()
        config.update(base_config)

        return config
