import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from polpo.ext.sklearn.compose import TransformedTargetRegressor
from polpo.ext.sklearn.truncation import truncate


def test_truncate_unsupported():
    with pytest.raises(TypeError, match="Truncation is not defined"):
        truncate(LinearRegression(), 2)


def test_truncate_composite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=(20, 4))

    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                TransformedTargetRegressor(
                    regressor=MultiOutputRegressor(LinearRegression()),
                    transformer=PCA(n_components=3),
                ),
            ),
        ]
    ).fit(X, y)

    truncated = truncate(estimator, 2)
    model = truncated.named_steps["regressor"]

    assert truncated is not estimator
    assert truncated.named_steps["scaler"] is estimator.named_steps["scaler"]
    assert model.transformer_.n_components_ == 2
    assert len(model.regressor_.estimators_) == 2
    assert estimator.named_steps["regressor"].transformer_.n_components_ == 3
    assert truncated.predict(X).shape == y.shape
