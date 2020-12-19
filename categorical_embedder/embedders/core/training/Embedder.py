from keras.models import load_model

from categorical_embedder.embedders.core.aux.custom_object_handler import prepare_custom_objects
from categorical_embedder.utils.io_utils import load_from_pickle


class Embedder(object):
    def __init__(self, embedder_model_path, artifacts_path):
        # Might consider loading model and artifacts from given path; but for now handle it outside.
        self._artifacts = self._load_artifacts(artifacts_path)
        self._model = self._load_embedder_model(embedder_model_path)

        self._preprocessor = self._artifacts["preprocessor"]

    def embed(self, data):
        # Processed data is always single dimensional numpy array when training is False.
        processed_data = self._preprocessor.process(data, training=False)
        embedding = self._model.predict(processed_data)
        # if result is a list, return 1st, otherwise return itself!
        if isinstance(embedding, list) or isinstance(embedding, tuple):
            return embedding[0]
        else:
            return embedding

    def _load_embedder_model(self, embedder_model_path):
        model = load_model(embedder_model_path, custom_objects=prepare_custom_objects(self._artifacts["custom_objects_info"]), compile=False)
        model.compile()
        return model

    def _load_artifacts(self, artifacts_path):
        return load_from_pickle(artifacts_path)
