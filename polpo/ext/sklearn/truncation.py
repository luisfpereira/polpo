"""Utilities for truncating fitted sklearn estimators.

This module defines a generic ``truncate`` operation for creating lower-rank
views of already fitted estimators without refitting them.

Truncation is implemented through type-specific registrations and supports
recursive composition for compatible sklearn estimators.
"""

from copy import copy
from functools import singledispatch

from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from polpo.sklearn.compose import TransformedTargetRegressor
from polpo.sklearn.decomposition import TruncatedPCA
from polpo.sklearn.multioutput import TruncatedMultiOutputRegressor


@singledispatch
def truncate(estimator, n_components):
    """Create a truncated view of an already fitted estimator.

    Parameters
    ----------
    estimator : estimator
        Fitted estimator to truncate.
    n_components : int
        Number of leading components or outputs to retain.

    Returns
    -------
    estimator
        Fitted estimator representing the truncated model.

    Raises
    ------
    TypeError
        If truncation is not defined for the estimator type.

    Notes
    -----
    Truncation does not refit the estimator. Implementations reuse fitted
    state from the original estimator and retain only the leading components
    or outputs required by the truncated model.

    Composite estimators may recursively truncate supported sub-estimators
    while leaving unsupported components unchanged.
    """
    raise TypeError(f"Truncation is not defined for {type(estimator).__name__}.")


def _supports_truncation(estimator):
    """Check whether truncation is defined for an estimator."""
    return truncate.dispatch(type(estimator)) is not truncate.dispatch(object)


def _truncate_if_supported(estimator, n_components):
    """Truncate an estimator when supported; otherwise return it unchanged."""
    if not _supports_truncation(estimator):
        return estimator

    return truncate(estimator, n_components)


@truncate.register(PCA)
def truncate_pca(estimator, n_components):
    """Truncate a fitted PCA estimator to its leading components."""
    return TruncatedPCA.from_fitted(estimator, n_components)


@truncate.register(MultiOutputRegressor)
def truncate_multi_output_regressor(estimator, n_components):
    """Truncate a fitted multi-output regressor to its leading outputs."""
    return TruncatedMultiOutputRegressor.from_fitted(
        estimator,
        n_outputs=n_components,
    )


@truncate.register(Pipeline)
def truncate_pipeline(estimator, n_components):
    """Recursively truncate supported steps of a fitted pipeline."""
    truncated = copy(estimator)

    truncated.steps = [
        (name, _truncate_if_supported(step, n_components))
        for name, step in estimator.steps
    ]

    return truncated


@truncate.register(TransformedTargetRegressor)
def truncate_transformed_target_regressor(estimator, n_components):
    """Truncate the fitted target transformer and regressor consistently."""
    truncated = copy(estimator)

    truncated.transformer_ = truncate(
        estimator.transformer_,
        n_components,
    )
    truncated.regressor_ = truncate(
        estimator.regressor_,
        n_components,
    )

    return truncated
