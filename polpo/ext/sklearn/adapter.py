"""Adapters for sklearn."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import available_if


def _has_inverse(adapter):
    return hasattr(adapter.transform, "inverse")


class TransformerAdapter(TransformerMixin, BaseEstimator):
    """Adapt a stateless callable to the sklearn transformer API.

    Parameters
    ----------
    transform : callable
        Transformation function, potentially exposing an ``inverse`` method.
    """

    def __init__(self, transform):
        self.transform = transform

    def __sklearn_clone__(self):
        """Return a new adapter wrapping the same transformation."""
        return TransformerAdapter(self.transform)

    def fit(self, X, y=None):
        """Return self without fitting."""
        return self

    @available_if(_has_inverse)
    def inverse_transform(self, X):
        """Apply the inverse transformation to X."""
        return self.transform.inverse(X)


def adapt_transform(transformer):
    """Adapt a callable to the scikit-learn transformer interface.

    Existing transformers implementing ``fit`` and ``transform`` are returned unchanged.

    Parameters
    ----------
    transformer : callable or transformer
        Transformation to adapt.

    Returns
    -------
    transformer : object
        Original transformer or a ``TransformerAdapter`` wrapping the callable.
    """
    if hasattr(transformer, "fit") and hasattr(transformer, "transform"):
        return transformer

    return TransformerAdapter(transformer)
