from abc import ABC, abstractmethod

from categorical_embedder.generators.InnerGenerator import InnerGenerator
from categorical_embedder.utils.io_utils import save_to_pickle


class TrainerABC(ABC):
    def __init__(self, preprocessor, outer_generator, generator_info, model_info, save_info):
        self._preprocessor = preprocessor
        self._data_generator = InnerGenerator(outer_generator, self._preprocessor, generator_info["pass_count"], generator_info["use_remaining"])
        self._model_info = model_info
        self._save_info = save_info

        # May allow more loss, also may represent it as a list of dict.
        self._main_model_artifacts = {"preprocessor": self._preprocessor, "custom_objects_info": {"loss_info": self._model_info["loss_info"], "has_implicit_loss": self._model_info["has_implicit_loss"]}}
        self._embedder_artifacts = {"preprocessor": self._preprocessor, "custom_objects_info": {"loss_info": self._model_info["loss_info"], "has_implicit_loss": self._model_info["has_implicit_loss"]}}

    @abstractmethod
    def train(self):
        # Made this abstract to as there may be different compile (e.g. add_loss usage..) or other choices.
        pass

    @abstractmethod
    def _get_model(self):
        pass

    @abstractmethod
    def _extract_embedder_model(self, model):
        pass

    def _save_models(self, model):
        model.save(self._save_info["main_model_path"])

        embedder_model = self._extract_embedder_model(model)
        embedder_model.save(self._save_info["embedder_model_path"])

    def _save_artifacts(self):
        save_to_pickle(self._main_model_artifacts, self._save_info["main_model_artifacts_path"])
        save_to_pickle(self._embedder_artifacts, self._save_info["embedder_artifacts_path"])
