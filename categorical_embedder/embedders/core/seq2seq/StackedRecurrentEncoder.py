from keras import backend as K
from keras.layers import Layer
from keras.layers.recurrent import SimpleRNN, GRU, LSTM


class StackedRecurrentEncoder(Layer):
    def __init__(self, units, num_of_layers, recurrent_type, recurrent_parameters, should_normalize, **kwargs):
        super().__init__(**kwargs)
        self._units = units
        self._num_of_layers = num_of_layers
        self._recurrent_type = recurrent_type
        self._recurrent_parameters = recurrent_parameters
        self._should_normalize = should_normalize
        self._recurrent_layers = []
        for _ in range(self._num_of_layers - 1):
            self._recurrent_layers.append(self._get_recurrent(True, False))
        self._recurrent_layers.append(self._get_recurrent(False, True))

    def call(self, inputs, **kwargs):
        output = K.cast_to_floatx(inputs)
        for i in range(self._num_of_layers - 1):
            output = self._recurrent_layers[i](output)

        result = self._recurrent_layers[-1](output)
        if self._should_normalize:
            result = [K.l2_normalize(t, axis=1) for t in result]

        return result[1:]

    def compute_output_shape(self, input_shape):
        if self._recurrent_type == "LSTM":
            return [(input_shape[0], self._units), (input_shape[0], self._units)]
        else:
            return [(input_shape[0], self._units)]

    def get_config(self):
        config = {
            "units": self._units,
            "num_of_layers": self._num_of_layers,
            "recurrent_type": self._recurrent_type,
            "recurrent_parameters": self._recurrent_parameters,
            "should_normalize": self._should_normalize
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
