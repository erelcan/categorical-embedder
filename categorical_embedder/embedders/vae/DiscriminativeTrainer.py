import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

from categorical_embedder.embedders.core.vae.VAETrainerABC import VAETrainerABC
from categorical_embedder.embedders.core.vae.concrete.VAE import VAE
from categorical_embedder.processors.DiscriminativeWrapper import DiscriminativeWrapper
from categorical_embedder.processors.HolisticCategoricalProcessor import HolisticCategoricalProcessor
from categorical_embedder.processors.SelfReturner import SelfReturner
from categorical_embedder.embedders.core.aux.loss_factory import get_loss_function


class DiscriminativeTrainer(VAETrainerABC):
    def __init__(self, num_of_categories, uniques, outer_generator, generator_info, model_info, save_info, discriminative_info):
        super().__init__(DiscriminativeWrapper(HolisticCategoricalProcessor(uniques, 0, np.int, shifter_on=False, mirror_target=True), SelfReturner()), outer_generator, generator_info, model_info, save_info)

        self._discriminative_info = discriminative_info

        self._num_of_categories = num_of_categories
        self._vocab_size = self._preprocessor.get_feature_processor().get_vocab_size()

    def _get_model(self):
        encoder_input = Input((self._num_of_categories, self._vocab_size), dtype="int64")
        enc_dec = VAE(self._num_of_categories, self._vocab_size, self._model_info["hidden_length"],
                      self._model_info["encoder_latent_info"], self._model_info["encoder_layer_info"],
                      self._model_info["decoder_layer_info"], self._model_info["inner_loss_info"], name="main")
        decoder_output, encoded = enc_dec(encoder_input)

        # dense_output = Dense(10, "relu")(encoded)
        # discriminative_output = Dense(self._discriminative_info["target_dim_length"], activation=self._discriminative_info["activation"], name="discriminative")(dense_output)

        discriminative_output = Dense(self._discriminative_info["target_dim_length"], activation=self._discriminative_info["activation"], name="discriminative")(encoded)

        model = Model(inputs=encoder_input, outputs=[decoder_output, discriminative_output])

        self._main_model_artifacts["custom_objects_info"]["layer_info"] = ["VAE", "VAEEncoder", "VAEDecoder"]
        self._embedder_artifacts["custom_objects_info"]["layer_info"] = ["VAEEncoder"]

        return model

    def _extract_embedder_model(self, model):
        encoder_input = model.layers[0].output
        embedding = model.get_layer("main").get_encoder()(encoder_input)
        embedder_model = Model(inputs=encoder_input, outputs=embedding)
        # Maybe misleading...
        # Consider not providing any info in compile~
        if self._model_info["has_implicit_loss"]:
            embedder_model.compile(optimizer=self._model_info["optimizer"], metrics=self._model_info["metrics"]["discriminative"])
        else:
            embedder_model.compile(optimizer=self._model_info["optimizer"], loss=get_loss_function(self._model_info["loss_info"]), metrics=self._model_info["metrics"]["discriminative"])

        return embedder_model
