import numpy as np
from kneebow.rotor import Rotor as _Rotor


class Rotor:
    def __init__(self):
        self._model = _Rotor()

    @property
    def elbow_index_(self):
        return self._model.get_elbow_index()

    def __getattr__(self, name):
        """Get attribute.

        It is only called when ``__getattribute__`` fails.
        Delegates attribute calling to _model.
        """
        return getattr(self._model, name)

    def fit(self, X, y):
        self._model.fit_rotate(np.stack([X, y]).T)

        return self
