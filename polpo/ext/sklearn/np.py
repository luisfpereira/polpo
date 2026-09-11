"""Transformer mixins handling numpy arrays."""

import numpy as np
from sklearn.preprocessing import FunctionTransformer

from polpo.preprocessing.np import FlattenButFirst, InvHstack, Reshape


class BiFlattenButFirst(FunctionTransformer):
    """Function transformer for for flattening array but first axis."""

    def __init__(self):
        super().__init__(
            func=FlattenButFirst(),
            inverse_func=Reshape(None),
            check_inverse=False,
        )

    def fit(self, X, y=None):
        # y is ignored
        self.inverse_func.shape = (-1,) + X.shape[1:]
        return self


class BiHstack(FunctionTransformer):
    def __init__(self):
        super().__init__(
            func=np.hstack, inverse_func=InvHstack(None), check_inverse=False
        )

    def fit(self, X, y=None):
        # y is ignored
        self.inverse_func.sizes = [x.shape[1] for x in X]
        return self
