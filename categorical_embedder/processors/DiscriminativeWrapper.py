from categorical_embedder.processors.ProcessorABC import ProcessorABC


class DiscriminativeWrapper(ProcessorABC):
    def __init__(self, feature_processor, label_processor):
        super().__init__()
        self._feature_processor = feature_processor
        self._label_processor = label_processor

    def process(self, data, training=True):
        if training:
            # data: [features: numpy ndarray, labels: numpy ndarray]
            # If features is a list or tuple, we will assume the last one is for target!
            # Re-consider and better design this.~
            if (isinstance(data, list) or isinstance(data, tuple)) and len(data) == 2:
                processed1 = self._feature_processor.process(data[0])
                processed2 = self._label_processor.process(data[1])
                if isinstance(processed1, list) or isinstance(processed1, tuple):
                    return processed1[0:-1], {"main": processed1[-1], "discriminative": processed2}
                else:
                    raise Exception("Data for DiscriminativeWrapper should have at least 2 target data: one for main embedding, and one for discriminative.")
            else:
                raise Exception("Data for DiscriminativeWrapper should be a list or tuple with length 2, for training.")
        else:
            # data: numpy ndarray
            return self._feature_processor.process(data, training=False)

    def get_feature_processor(self):
        return self._feature_processor

    def get_label_processor(self):
        return self._label_processor
