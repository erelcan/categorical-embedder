from copy import deepcopy
import numpy as np

from categorical_embedder.processors.ProcessorABC import ProcessorABC


# Only for 2D and 3D data...
class SelfOrExpandReturner(ProcessorABC):
    def __init__(self, copy_on=False, mirror_target=False):
        super().__init__()
        self._copy_on = copy_on
        self._mirror_target = mirror_target

    def process(self, data, training=True):
        if len(data.shape) == 3:
            processed_data = deepcopy(data) if self._copy_on else data
        else:
            processed_data = np.expand_dims(data, axis=-1)

        if training:
            if self._mirror_target:
                return processed_data, deepcopy(processed_data)
            else:
                return processed_data, None
        else:
            return processed_data
