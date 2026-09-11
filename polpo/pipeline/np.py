"""Preprocessing steps involving numpy arrays."""

import numpy as np

from polpo.preprocessing.base import PreprocessingStep


class Stack(PreprocessingStep):
    """Joins a sequence of arrays along a new axis.

    https://numpy.org/doc/stable/reference/generated/numpy.stack.html

    Parameters
    ----------
    axis : int
        The axis in the result array along which the input arrays are stacked.
    """

    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def __call__(self, data):
        return np.stack(data, axis=self.axis)


class Reshape(PreprocessingStep):
    """Gives a new shape to an array without changing its data.

    https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy-reshape

    Parameters
    ----------
    shape : int or tuple[int]
        New shape. Must be compatible.
    """

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __call__(self, data):
        return np.reshape(data, self.shape)


class FlattenButFirst(PreprocessingStep):
    """Flattens array but first axis."""

    def __call__(self, data):
        return np.reshape(data, (data.shape[0], -1))


class InvHstack(PreprocessingStep):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes

    def __call__(self, data):
        cumsize = np.r_[[0], np.cumsum(self.sizes)]
        return [
            data[:, init_index:last_index]
            for init_index, last_index in zip(cumsize, cumsize[1:])
        ]


class Complex2RealsRepr(PreprocessingStep):
    """Transforms a complex number into a 2-vector."""

    def __call__(self, cnumber):
        return np.hstack([cnumber.real, cnumber.imag])


class RealsRepr2Complex(PreprocessingStep):
    """Transforms a 2-vector into a complex number."""

    def __call__(self, rnumber):
        half_index = rnumber.shape[-1] // 2
        return rnumber[..., :half_index] + 1j * rnumber[..., half_index:]


class ConcatenationIndices(PreprocessingStep):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def __call__(self, arrays):
        start_index = 0
        indices = []
        for array in arrays:
            dim = array.shape[self.axis]
            end_index = start_index + dim
            indices.append((start_index, end_index))
            start_index = end_index

        return indices
