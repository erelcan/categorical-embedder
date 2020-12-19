from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers import Dense, Conv1D, MaxPooling1D, UpSampling1D, TimeDistributed, RepeatVector, Flatten

from categorical_embedder.embedders.core.aux.SelectorLayer import SelectorLayer
from categorical_embedder.embedders.core.aux.ExpandLayer import ExpandLayer
from categorical_embedder.embedders.core.seq2seq.StackedRecurrentEncoder import StackedRecurrentEncoder
from categorical_embedder.embedders.core.seq2seq.StackedRecurrentDecoder import StackedRecurrentDecoder


def create(layer_type, parameters):
    return _allowed_layers[layer_type](parameters)


_allowed_layers = {
    "RNN": lambda parameters: SimpleRNN(**parameters),
    "GRU": lambda parameters: GRU(**parameters),
    "LSTM": lambda parameters: LSTM(**parameters),
    "Conv1D": lambda parameters: Conv1D(**parameters),
    "MaxPooling1D": lambda parameters: MaxPooling1D(**parameters),
    "UpSampling1D": lambda parameters: UpSampling1D(**parameters),
    "Dense": lambda parameters: Dense(**parameters),
    "TDD": lambda parameters: TimeDistributed(Dense(**parameters)),
    "RepeatVector": lambda parameters: RepeatVector(**parameters),
    "Flatten": lambda parameters: Flatten(**parameters),
    "SelectorLayer": lambda parameters: SelectorLayer(**parameters),
    "ExpandLayer": lambda parameters: ExpandLayer(**parameters),
    "StackedRecurrentEncoder": lambda parameters: StackedRecurrentEncoder(**parameters),
    "StackedRecurrentDecoder": lambda parameters: StackedRecurrentDecoder(**parameters)
}
