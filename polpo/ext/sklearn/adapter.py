"""Adapters for sklearn."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import available_if


def _has_inverse(adapter):
    return hasattr(adapter.transform, "inverse")


class TransformerAdapter(TransformerMixin, BaseEstimator):
    """Adapt a stateless callable to the sklearn transformer API."""

    def __init__(self, transform):
        self.transform = transform

    def __sklearn_clone__(self):
        return TransformerAdapter(self.transform)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.transform(X)

    @available_if(_has_inverse)
    def inverse_transform(self, X):
        return self.transform.inverse(X)


def adapt_transform(transformer):
    if hasattr(transformer, "fit") and hasattr(transformer, "transform"):
        return transformer

    return TransformerAdapter(transformer)
