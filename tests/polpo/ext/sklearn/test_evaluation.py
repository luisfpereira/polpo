import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from polpo.ext.sklearn.evaluation import (
    TruncatedCVEvaluationResult,
    TruncatedCVEvaluator,
)


@pytest.mark.smoke
def test_truncated_cv_evaluator():
    x = np.tile(np.arange(4), 3)
    X = x[:, None]
    y = np.column_stack([x, 2 * x])
    groups = np.repeat(np.arange(3), 4)
    keys = list(range(len(x)))

    evaluator = TruncatedCVEvaluator(
        estimator=MultiOutputRegressor(LinearRegression()),
        truncations=[1, 2],
        metrics={"mse": lambda true, pred: np.mean((true[: len(pred)] - pred) ** 2)},
        prepare_data=lambda data: data,
        n_jobs=1,
    )

    assert evaluator.fit((X, y, groups, keys)) is evaluator

    result = evaluator.result_

    assert result.keys == keys
    assert result.truncations == [1, 2]
    assert result.held_out_groups == [0, 1, 2]
    np.testing.assert_allclose(result.metrics["mse"], 0, atol=1e-24)


def test_truncated_cv_evaluation_result_io(tmp_path):
    result = TruncatedCVEvaluationResult(
        metrics={"mse": np.array([[1, 2], [3, 4]])},
        keys=["a", "b"],
        truncations=[1, 2],
        held_out_groups=[0, 1],
        fit_diagnostics={"score": 0.5},
    )

    assert result.to_dir(tmp_path) is result

    loaded = TruncatedCVEvaluationResult.from_dir(tmp_path)

    np.testing.assert_array_equal(loaded.metrics["mse"], result.metrics["mse"])
    assert loaded.keys == result.keys
    assert loaded.truncations == result.truncations
    assert loaded.held_out_groups == result.held_out_groups
    assert loaded.fit_diagnostics == result.fit_diagnostics
