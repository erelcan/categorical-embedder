from copy import deepcopy

from categorical_embedder.processors.ProcessorABC import ProcessorABC


class SelfReturner(ProcessorABC):
    def __init__(self, copy_on=False):
        super().__init__()
        self._copy_on = copy_on

    def process(self, data, **kwargs):
        return deepcopy(data) if self._copy_on else data
