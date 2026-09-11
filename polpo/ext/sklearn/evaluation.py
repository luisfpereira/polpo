"""Cross-validated evaluation of truncated estimators."""

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from polpo.io.json import load_json, save_json
from polpo.sklearn.model_selection import (
    assemble_predictions,
    cross_fit,
    predict_folds,
)
from polpo.sklearn.truncation import truncate


class TruncatedCVEvaluator:
    """Evaluate predictions across truncations of a fitted estimator.

    Parameters
    ----------
    estimator : estimator
        Maximal estimator to fit in each cross-validation fold.
    truncations : iterable of int
        Truncation levels to evaluate.
    metrics : dict
        Mapping from metric names to callables accepting a target and a
        prediction.
    prepare_data : callable
        Function mapping the input dataset to ``X, y, groups, keys``.
    fit_diagnostics : callable, optional
        Function computing diagnostics from the cross-fitting result.
    cv : cross-validation splitter, optional
        Cross-validation splitting strategy. Defaults to
        ``LeaveOneGroupOut``.
    n_jobs : int, optional
        Number of jobs used during cross-fitting.
    """

    def __init__(
        self,
        estimator,
        truncations,
        metrics,
        prepare_data,
        fit_diagnostics=None,
        cv=None,
        n_jobs=None,
    ):
        if cv is None:
            cv = LeaveOneGroupOut()

        self.estimator = estimator
        self.truncations = list(truncations)
        self.metrics = metrics
        self.prepare_data = prepare_data
        self.fit_diagnostics = fit_diagnostics
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, dataset):
        """Fit cross-validation models and evaluate their truncations."""
        X, y, groups, keys = self.prepare_data(dataset)

        self.cross_fit_result_ = cross_fit(
            self.estimator,
            X,
            y,
            groups=groups,
            cv=self.cv,
            n_jobs=self.n_jobs,
        )

        held_out_groups = [
            np.unique(groups[test_indices]).item()
            for test_indices in self.cross_fit_result_.test_indices
        ]

        fit_diagnostics = None
        if self.fit_diagnostics is not None:
            fit_diagnostics = self.fit_diagnostics(self.cross_fit_result_)

        distances = self._evaluate(
            self.cross_fit_result_,
            X,
            y,
        )

        self.result_ = TruncatedCVEvaluationResult(
            distances=distances,
            keys=keys,
            truncations=self.truncations,
            held_out_groups=held_out_groups,
            fit_diagnostics=fit_diagnostics,
        )

        return self

    def _evaluate(self, cross_fit_result, X, y):
        distances = {
            name: np.empty((len(y), len(self.truncations))) for name in self.metrics
        }

        for truncation_idx, truncation in enumerate(self.truncations):
            predictions = self._predict(
                cross_fit_result,
                X,
                truncation,
            )

            for name, metric in self.metrics.items():
                distances[name][:, truncation_idx] = [
                    metric(y_true, y_pred) for y_true, y_pred in zip(y, predictions)
                ]

        return distances

    @staticmethod
    def _predict(cross_fit_result, X, truncation):
        estimators = [
            truncate(estimator, truncation) for estimator in cross_fit_result.estimators
        ]

        predictions = predict_folds(
            estimators,
            X,
            cross_fit_result.test_indices,
        )

        return assemble_predictions(
            predictions,
            cross_fit_result.test_indices,
        )


class TruncatedCVEvaluationResult:
    """Results of cross-validated truncated-estimator evaluation.

    Parameters
    ----------
    distances : dict
        Mapping from metric names to arrays of shape
        ``(n_samples, n_truncations)``.
    keys : list
        Keys identifying samples represented by rows of ``distances``.
    truncations : list
        Evaluated truncation levels.
    held_out_groups : list
        Group held out in each cross-validation fold.
    fit_diagnostics : object
        Diagnostics computed from the fitted fold estimators.
    """

    def __init__(
        self,
        distances,
        keys,
        truncations,
        held_out_groups,
        fit_diagnostics=None,
    ):
        self.distances = distances
        self.keys = list(keys)
        self.truncations = list(truncations)
        self.held_out_groups = list(held_out_groups)
        self.fit_diagnostics = fit_diagnostics

    def to_dir(self, results_dir):
        """Write results to disk."""
        results_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            results_dir / "distances.npz",
            **self.distances,
        )

        save_json(
            results_dir / "params.json",
            {
                "keys": self.keys,
                "truncations": self.truncations,
                "held_out_groups": self.held_out_groups,
            },
        )

        if self.fit_diagnostics is not None:
            save_json(
                results_dir / "fit_diagnostics.json",
                self.fit_diagnostics,
            )

        return self

    @classmethod
    def from_dir(cls, results_dir):
        """Load results from disk."""
        with np.load(results_dir / "distances.npz") as data:
            distances = {name: values.copy() for name, values in data.items()}

        params = load_json(results_dir / "params.json")

        diagnostics_path = results_dir / "fit_diagnostics.json"
        fit_diagnostics = (
            load_json(diagnostics_path) if diagnostics_path.exists() else None
        )

        return cls(
            distances=distances,
            fit_diagnostics=fit_diagnostics,
            **params,
        )
