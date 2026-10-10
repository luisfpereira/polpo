import numpy as np
import pandas as pd

from polpo.ext.sklearn.regression import MixedLMRegressor


def test_mixedlm_regressor():
    rng = np.random.default_rng(0)
    x = np.tile(np.linspace(-1, 1, 8), 5)
    groups = np.repeat(np.arange(5), 8)

    X = pd.DataFrame({"x": x, "subject": groups})
    y = 2 + 3 * x + np.repeat([-1, -0.5, 0, 0.5, 1], 8)
    y += rng.normal(scale=0.03, size=len(x))

    model = MixedLMRegressor("Y ~ x", groups="subject")
    assert model.fit(X, y) is model

    predictions = model.predict(pd.DataFrame({"x": [0, 1]}))

    np.testing.assert_allclose(predictions, [2, 5], atol=0.1)
    assert "Y" not in X.columns
