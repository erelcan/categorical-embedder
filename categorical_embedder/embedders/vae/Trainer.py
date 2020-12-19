import numpy as np
from keras.layers import Input
from keras.models import Model

from categorical_embedder.embedders.core.vae.VAETrainerABC import VAETrainerABC
from categorical_embedder.embedders.core.vae.concrete.VAE import VAE
from categorical_embedder.processors.HolisticCategoricalProcessor import HolisticCategoricalProcessor


class Trainer(VAETrainerABC):
    def __init__(self, num_of_categories, uniques, outer_generator, generator_info, model_info, save_info):
        super().__init__(HolisticCategoricalProcessor(uniques, 0, np.int, shifter_on=False), outer_generator, generator_info, model_info, save_info)

        self._num_of_categories = num_of_categories
        self._vocab_size = self._preprocessor.get_vocab_size()

    def _get_model(self):
        encoder_input = Input((self._num_of_categories, self._vocab_size), dtype="int64")
        enc_dec = VAE(self._num_of_categories, self._vocab_size, self._model_info["hidden_length"], self._model_info["encoder_latent_info"], self._model_info["encoder_layer_info"], self._model_info["decoder_layer_info"], self._model_info["inner_loss_info"])
        decoder_output, _ = enc_dec(encoder_input)
        model = Model(inputs=encoder_input, outputs=decoder_output)

        self._main_model_artifacts["custom_objects_info"]["layer_info"] = ["VAE", "VAEEncoder", "VAEDecoder"]
        self._embedder_artifacts["custom_objects_info"]["layer_info"] = ["VAEEncoder"]

        return model
