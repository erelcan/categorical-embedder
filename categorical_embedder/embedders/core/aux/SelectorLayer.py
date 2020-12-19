from keras.layers import Layer


class SelectorLayer(Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self._index = index

    def call(self, inputs, **kwargs):
        return inputs[self._index]

    def compute_output_shape(self, input_shape):
        return input_shape[self._index]

    def get_config(self):
        config = {
            "index": self._index
        }
        base_config = super().get_config()
        config.update(base_config)

        return config

    def get_encoder(self):
        return self._encoder
