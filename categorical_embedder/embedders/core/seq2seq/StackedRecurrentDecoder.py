from keras import backend as K
from keras.layers import Layer
from keras.layers.recurrent import SimpleRNN, GRU, LSTM


class StackedRecurrentDecoder(Layer):
    def __init__(self, units, num_of_layers, recurrent_type, recurrent_parameters, **kwargs):
        super().__init__(**kwargs)
        self._units = units
        self._num_of_layers = num_of_layers
        self._recurrent_type = recurrent_type
        self._recurrent_parameters = recurrent_parameters
        self._recurrent_layers = []
        for _ in range(self._num_of_layers):
            self._recurrent_layers.append(self._get_recurrent(True, False))

    def call(self, inputs, initial_state=None, **kwargs):
        output = self._recurrent_layers[0](K.cast_to_floatx(inputs), initial_state=initial_state)
        for i in range(1, self._num_of_layers):
            output = self._recurrent_layers[i](output)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self._units,)

    def get_config(self):
        config = {
            "units": self._units,
            "num_of_layers": self._num_of_layers,
            "recurrent_type": self._recurrent_type,
            "recurrent_parameters": self._recurrent_parameters
        }
        base_config = super().get_config()
        config.update(base_config)

        return config

    def _get_recurrent(self, return_sequences, return_state):
        params = {"units": self._units, "return_sequences": return_sequences, "return_state": return_state}
        params.update(self._recurrent_parameters)
        return _recurrent_creators[self._recurrent_type](params)


_recurrent_creators = {
    "RNN": lambda parameters: SimpleRNN(**parameters),
    "GRU": lambda parameters: GRU(**parameters),
    "LSTM": lambda parameters: LSTM(**parameters)
}
