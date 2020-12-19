import numpy as np
from keras.layers import Input
from keras.models import Model

from categorical_embedder.embedders.core.seq2seq.Seq2SeqTrainerABC import Seq2SeqTrainerABC
from categorical_embedder.embedders.core.seq2seq.EncoderDecoder import EncoderDecoder
from categorical_embedder.processors.HolisticCategoricalProcessor import HolisticCategoricalProcessor


class Trainer(Seq2SeqTrainerABC):
    def __init__(self, num_of_categories, uniques, outer_generator, generator_info, model_info, save_info):
        super().__init__(HolisticCategoricalProcessor(uniques, 0, np.int), outer_generator, generator_info, model_info, save_info)

        self._num_of_categories = num_of_categories
        self._vocab_size = self._preprocessor.get_vocab_size()

    def _get_model(self):
        encoder_input = Input((self._num_of_categories, self._vocab_size), dtype="int64")
        decoder_input = Input((self._num_of_categories + 1, self._vocab_size), dtype="int64")

        enc_dec = EncoderDecoder(self._vocab_size, self._model_info["embedding_length"], "softmax" if self._model_info["softmax_on"] else "linear", self._model_info["encoder_info"], self._model_info["decoder_info"])

        decoder_output, _ = enc_dec([encoder_input, decoder_input])
        model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)

        self._main_model_artifacts["custom_objects_info"]["layer_info"] = ["EncoderDecoder", "StackedRecurrentEncoder", "StackedRecurrentDecoder"]
        self._embedder_artifacts["custom_objects_info"]["layer_info"] = ["StackedRecurrentEncoder"]

        return model
