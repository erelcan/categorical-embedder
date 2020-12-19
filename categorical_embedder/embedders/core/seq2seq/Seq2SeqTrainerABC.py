from abc import abstractmethod
from keras.models import Model

from categorical_embedder.embedders.core.training.TrainerABC import TrainerABC
from categorical_embedder.embedders.core.aux.loss_factory import get_loss_function


# _extract_embedder_model knows the structure of the model!!
# _get_model is for filling custom parameters to a known structure.
# Re-consider for moving _extract_embedder_model to extending trainer and moving train() to super..
class Seq2SeqTrainerABC(TrainerABC):
    def __init__(self, preprocessor, outer_generator, generator_info, model_info, save_info):
        super().__init__(preprocessor, outer_generator, generator_info, model_info, save_info)

    def train(self):
        model = self._get_model()
        loss_weights = self._model_info.get("loss_weights", None)
        if self._model_info["has_implicit_loss"]:
            model.compile(optimizer=self._model_info["optimizer"], metrics=self._model_info["metrics"], loss_weights=loss_weights)
        else:
            model.compile(optimizer=self._model_info["optimizer"], loss=get_loss_function(self._model_info["loss_info"]), metrics=self._model_info["metrics"], loss_weights=loss_weights)

        model.summary()
        model.fit_generator(self._data_generator.get_generator(), steps_per_epoch=self._data_generator.get_num_of_steps(), **self._model_info["fit_parameters"])

        self._save_models(model)
        self._save_artifacts()

        return self._extract_embedder_model(model), model

    @abstractmethod
    def _get_model(self):
        pass

    def _extract_embedder_model(self, model):
        encoder_input = model.layers[0].output
        embedding = model.layers[-1].get_encoder()(encoder_input)
        embedder_model = Model(inputs=encoder_input, outputs=embedding)
        # Maybe misleading...
        # Consider not providing any info in compile~
        if self._model_info["has_implicit_loss"]:
            embedder_model.compile(optimizer=self._model_info["optimizer"], metrics=self._model_info["metrics"])
        else:
            embedder_model.compile(optimizer=self._model_info["optimizer"], loss=get_loss_function(self._model_info["loss_info"]), metrics=self._model_info["metrics"])

        return embedder_model
