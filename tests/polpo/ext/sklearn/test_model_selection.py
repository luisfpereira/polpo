import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut

from polpo.ext.sklearn.model_selection import (
    assemble_predictions,
    cross_fit,
    predict_folds,
)


def test_cross_fit():
    x = np.tile(np.arange(4), 3)
    X = x[:, None]
    y = 2 * x + 1
    groups = np.repeat(np.arange(3), 4)

    estimator = LinearRegression()
    result = cross_fit(estimator, X, y, groups=groups, cv=LeaveOneGroupOut(), n_jobs=1)

    assert len(result.estimators) == 3
    assert all(est is not estimator for est in result.estimators)
    assert all(len(indices) == 8 for indices in result.train_indices)

    predictions = predict_folds(result.estimators, X, result.test_indices)
    assembled = assemble_predictions(predictions, result.test_indices)

    np.testing.assert_allclose(assembled, y)


def test_assemble_predictions():
    predictions = [np.array([30, 10]), np.array([20])]
    indices = [np.array([2, 0]), np.array([1])]

    np.testing.assert_array_equal(
        assemble_predictions(predictions, indices),
        [10, 20, 30],
    )

    with pytest.raises(ValueError, match="duplicates"):
        assemble_predictions(predictions, [np.array([0, 0]), np.array([1])])

    with pytest.raises(ValueError, match="complete partition"):
        assemble_predictions(predictions, [np.array([0, 2]), np.array([3])])
