from keras import backend as K
from keras.layers import Layer


class ExpandLayer(Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self._axis = axis

    def call(self, inputs, **kwargs):
        return K.expand_dims(inputs, axis=self._axis)

    def compute_output_shape(self, input_shape):
        if self._axis < 0:
            axis = self._axis + len(input_shape) + 1
        else:
            axis = self._axis

        output_shape = input_shape[:axis] + (1,) + input_shape[axis:]

        return tuple(output_shape)

    def get_config(self):
        config = {
            "index": self._index
        }
        base_config = super().get_config()
        config.update(base_config)

        return config

    def get_encoder(self):
        return self._encoder
