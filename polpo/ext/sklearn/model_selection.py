"""Extensions for sklearn model-selection workflows.

This module provides utilities for fitting estimators across cross-validation
splits and for organizing fold-wise predictions without requiring scoring.

It complements sklearn's model-selection API with access to fitted fold
estimators, split indices, and reusable prediction assembly.
"""

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone, is_classifier
from sklearn.model_selection import check_cv
from sklearn.utils import _safe_indexing, indexable


@dataclass
class CrossFitResult:
    """Result of fitting an estimator across cross-validation splits."""

    estimators: list
    train_indices: list
    test_indices: list


def _fit_on_split(estimator, X, y, train_indices):
    estimator = clone(estimator)

    X_train = _safe_indexing(X, train_indices)

    if y is None:
        estimator.fit(X_train)
    else:
        y_train = _safe_indexing(y, train_indices)
        estimator.fit(X_train, y_train)

    return estimator


def cross_fit(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    cv=None,
    n_jobs=None,
    pre_dispatch="2*n_jobs",
):
    """Fit cloned estimators across cross-validation splits.

    Parameters
    ----------
    estimator : estimator
        Estimator implementing ``fit``.
    X : array-like
        Input data.
    y : array-like
        Target data.
    groups : array-like
        Group labels used by group-aware cross-validation splitters.
    cv : int, cross-validation splitter or iterable
        Cross-validation splitting strategy.
    n_jobs : int
        Number of jobs used to fit splits in parallel.
    pre_dispatch : int or str, default="2*n_jobs"
        Number of jobs dispatched during parallel execution.

    Returns
    -------
    result : CrossFitResult
        Fitted estimators and the corresponding train and test indices.
    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(
        cv,
        y,
        classifier=is_classifier(estimator),
    )

    splits = list(cv.split(X, y, groups))

    estimators = Parallel(
        n_jobs=n_jobs,
        pre_dispatch=pre_dispatch,
    )(
        delayed(_fit_on_split)(estimator, X, y, train_indices)
        for train_indices, _ in splits
    )

    return CrossFitResult(
        estimators=estimators,
        train_indices=[train for train, _ in splits],
        test_indices=[test for _, test in splits],
    )


def predict_folds(estimators, X, indices, method="predict"):
    """Predict independently on data associated with each fitted estimator.

    Parameters
    ----------
    estimators : iterable of estimators
        Fitted estimators, one per cross-validation fold.
    X : array-like
        Input data.
    indices : iterable of index arrays
        Sample indices associated with each estimator.
    method : str, default="predict"
        Estimator method used to generate predictions.

    Returns
    -------
    predictions : list
        Predictions for each fold, in the same order as ``estimators``.
    """
    return [
        getattr(estimator, method)(_safe_indexing(X, fold_indices))
        for estimator, fold_indices in zip(estimators, indices)
    ]


def assemble_predictions(predictions, indices):
    """Assemble fold predictions in the original sample order.

    Parameters
    ----------
    predictions : iterable of array-like
        Predictions generated independently for each fold.
    indices : iterable of index arrays
        Original sample indices corresponding to each prediction block.

    Returns
    -------
    assembled : array-like
        Predictions reordered to match the original sample order.

    Raises
    ------
    ValueError
        If the indices do not form a partition of the samples.
    """
    indices = np.concatenate(indices)

    if len(np.unique(indices)) != len(indices):
        raise ValueError("Indices must not contain duplicates.")

    order = np.argsort(indices)

    if not np.array_equal(indices[order], np.arange(len(indices))):
        raise ValueError("Indices must form a complete partition.")

    predictions = np.concatenate(
        [np.asarray(prediction) for prediction in predictions],
        axis=0,
    )

    return predictions[order]
