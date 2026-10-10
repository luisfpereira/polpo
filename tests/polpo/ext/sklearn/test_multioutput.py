import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from polpo.ext.sklearn.multioutput import TruncatedMultiOutputRegressor


def test_truncated_multi_output_regressor():
    X = np.arange(10).reshape(-1, 1)
    y = np.column_stack([X[:, 0], 2 * X[:, 0], 3 * X[:, 0]])

    regressor = MultiOutputRegressor(LinearRegression()).fit(X, y)
    truncated = TruncatedMultiOutputRegressor.from_fitted(regressor, n_outputs=2)

    assert len(truncated.estimators_) == 2
    assert len(regressor.estimators_) == 3
    assert truncated.estimators_[0] is regressor.estimators_[0]

    np.testing.assert_allclose(
        truncated.predict(X),
        regressor.predict(X)[:, :2],
    )

    with pytest.raises(ValueError):
        TruncatedMultiOutputRegressor.from_fitted(regressor, n_outputs=4)
