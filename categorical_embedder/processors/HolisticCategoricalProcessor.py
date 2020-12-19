import numpy as np

from categorical_embedder.processors.ProcessorABC import ProcessorABC
from categorical_embedder.processors.Seq2SeqShifter import Seq2SeqShifter


class HolisticCategoricalProcessor(ProcessorABC):
    def __init__(self, uniques, start_val, target_type, shifter_on=True, mirror_target=False):
        super().__init__()
        self._uniques = uniques
        self._start_val = start_val
        self._target_type = target_type
        self._shifter_on = shifter_on
        self._mirror_target = mirror_target

        self._mapper, vocab_size = self._create_mapper()
        self._start_token = self._start_val + vocab_size
        self._end_token = self._start_token + 1
        self._vocab_size = vocab_size + 2

        self._shifter = Seq2SeqShifter(self._start_token, self._end_token)

    def process(self, data, training=True):
        # Caution: Modifying data..
        mapped_data = np.zeros(data.shape, dtype=self._target_type)
        for col in self._uniques:
            for i in range(data.shape[0]):
                mapped_data[i][col] = self._map_domain(data[i][col], col)

        if training:
            if self._shifter_on:
                encoder_input, decoder_input, decoder_target = self._shifter.process(mapped_data)
                return [onehot_initialization(encoder_input, self._vocab_size), onehot_initialization(decoder_input, self._vocab_size)], onehot_initialization(decoder_target, self._vocab_size)
            else:
                if self._mirror_target:
                    return onehot_initialization(mapped_data, self._vocab_size), onehot_initialization(mapped_data, self._vocab_size)
                else:
                    return onehot_initialization(mapped_data, self._vocab_size), None
        else:
            # As we know that the shifter will keep the encoder input same when aux_tokens are provided.
            return onehot_initialization(mapped_data, self._vocab_size)

    def get_vocab_size(self):
        return self._vocab_size

    def _map_domain(self, val, col):
        if (isinstance(val, str) or isinstance(val, int) or isinstance(val, float)) and val in self._mapper[col]:
            return self._mapper[col][val]
        else:
            return self._mapper[col]["other"]

    def _create_mapper(self):
        # key in uniques must refer to the id of the corresponding column in the numpy array!
        last_val = self._start_val
        mapping = {}
        for key in self._uniques:
            mapping[key] = {"other": last_val}
            last_val += 1
            for v in self._uniques[key]:
                if isinstance(v, str) or isinstance(v, int) or isinstance(v, float):
                    mapping[key][v] = last_val

                last_val += 1

        return mapping, last_val - self._start_val


def onehot_initialization(a, vocab_size):
    ncols = vocab_size
    out = np.zeros(a.shape + (ncols,), dtype=int)
    out[all_idx(a, axis=2)] = 1
    return out


def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)
