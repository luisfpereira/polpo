import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from polpo.ext.sklearn.adapter import TransformerAdapter, adapt_transform


def test_transformer_adapter():
    def shift(X):
        return X + 1

    shift.inverse = lambda X: X - 1

    X = np.array([[1, 2], [3, 4]])
    adapter = adapt_transform(shift)

    assert isinstance(adapter, TransformerAdapter)
    np.testing.assert_array_equal(adapter.fit_transform(X), X + 1)
    np.testing.assert_array_equal(adapter.inverse_transform(X + 1), X)

    assert clone(adapter) is not adapter


def test_adapt_transform_existing():
    transformer = StandardScaler()

    assert adapt_transform(transformer) is transformer
