import numpy as np
from copy import deepcopy

from categorical_embedder.processors.ProcessorABC import ProcessorABC


# Designed only for structured embedding.
# Therefore, does not consider padding!
class Seq2SeqShifter(ProcessorABC):
    def __init__(self, start_token=None, end_token=None):
        super().__init__()
        self._start_token = start_token
        self._end_token = end_token

    def process(self, data, **kwargs):
        if self._start_token is not None and self._end_token is not None:
            return self._handle_with_aux_tokens(data)
        else:
            return self._handle_without_aux_tokens(data)

    def _handle_with_aux_tokens(self, data):
        if len(data.shape) == 2:
            start_slice = np.ones((data.shape[0], 1), dtype=data.dtype) * self._start_token
            end_slice = np.ones((data.shape[0], 1), dtype=data.dtype) * self._end_token
        else:
            start_slice = np.ones((data.shape[0], 1, data.shape[2]), dtype=data.dtype) * self._start_token
            end_slice = np.ones((data.shape[0], 1, data.shape[2]), dtype=data.dtype) * self._end_token

        # Be careful on data copies and references!
        # The changes in decoder_input and decoder_target does not effect encoder_input and vice-a-versa.
        # However, test in detail~
        encoder_input = data
        decoder_input = np.concatenate((start_slice, data), axis=1)
        decoder_target = np.concatenate((data, end_slice), axis=1)

        return encoder_input, decoder_input, decoder_target

    def _handle_without_aux_tokens(self, data):
        # Sequence length should be at least 3!
        # Re-consider deepcopy, maybe make it optional!
        if len(data.shape) == 2:
            encoder_input = deepcopy(data[:, 1:-1])
            decoder_input = deepcopy(data[:, 0:-1])
            decoder_target = deepcopy(data[:, 1:])
        else:
            encoder_input = deepcopy(data[:, 1:-1, :])
            decoder_input = deepcopy(data[:, 0:-1, :])
            decoder_target = deepcopy(data[:, 1:, :])

        return encoder_input, decoder_input, decoder_target
