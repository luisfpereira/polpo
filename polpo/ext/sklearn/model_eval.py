import abc
from collections.abc import Iterable

import numpy as np
import scipy
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.metrics import r2_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.validation import check_is_fitted
from statsmodels.stats.multitest import multipletests

from polpo.sklearn.adapter import EvaluatedModel

from .adapter import TransformerAdapter


class ModelEvaluator:
    def __init__(self, prefix="", extender=None):
        # TODO: rename extender
        if extender is None:
            extender = lambda x: x

        self.prefix = prefix
        self.extender = extender

    @abc.abstractmethod
    def __call__(self, model, X, y=None, y_pred=None):
        pass

    def _prefix_keys(self, results):
        if self.prefix:
            return dict(
                (f"{self.prefix}-{key}", value) for (key, value) in results.items()
            )

        return results

    def _update_results(self, results):
        return self._prefix_keys(self.extender(results))


class MultiEvaluator(ModelEvaluator):
    def __init__(self, evaluators, extender=None):
        super().__init__("", extender)
        self.evaluators = evaluators

    def __call__(self, model, X, y=None, y_pred=None):
        # assumes X does not contain intercept

        if hasattr(model, "predict") and y_pred is None:
            y_pred = model.predict(X)

        results = {}
        for evaluator in self.evaluators:
            results_ = evaluator(model, X, y, y_pred)
            results.update(results_)

        return self._update_results(results)


class AdapterPipeline(Pipeline):
    """sklearn compatible pipeline.

    Names and adapts steps if needed.
    Syntax sugar for `sklearn.Pipeline` without the
    need to name steps, and with the ability of having
    callables as steps.

    Parameters
    ----------
    steps : list
        Steps to be adapted.
    """

    # TODO: remove

    def __init__(self, steps):
        self._unadapted_steps = steps

        adapted_steps = []
        for index, step in enumerate(steps):
            if not isinstance(step, Iterable):
                step_name = f"step_{index}"
                step = (step_name, step)

            if not hasattr(step[1], "fit"):
                step = (step[0], TransformerAdapter(step[1]))

            adapted_steps.append(step)

        super().__init__(steps=adapted_steps)

    def __sklearn_clone__(self):
        return AdapterPipeline(steps=self._unadapted_steps)

    def predict_eval(self, X, y=None):
        check_is_fitted(self)

        Xt = X
        for _, _, transform in self._iter(with_final=False):
            if hasattr(transform, "transform_eval"):
                Xt = transform.transform_eval(Xt, y)
            else:
                Xt = transform.transform(Xt)

            if isinstance(Xt, tuple):
                Xt, _ = Xt

        last_step = self.steps[-1][1]
        if hasattr(last_step, "predict_eval"):
            return last_step.predict_eval(Xt, y)

        return last_step.predict(Xt)

    def transform_eval(self, X, y=None):
        check_is_fitted(self)

        Xt = X
        for _, _, transform in self._iter():
            # TODO: need to check if last has transform?
            if hasattr(transform, "transform_eval"):
                Xt = transform.transform_eval(Xt, y)
            else:
                Xt = transform.transform(Xt)

        return Xt


class AdapterFeatureUnion(FeatureUnion):
    def __init__(
        self,
        transformer_list,
        *,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True,
    ):
        self._unadapted_transformer_list = transformer_list

        adapted_transformer_list = []
        for index, transformer in enumerate(transformer_list):
            if not isinstance(transformer, Iterable):
                transformer_name = f"step_{index}"
                transformer = (transformer_name, transformer)

            if not hasattr(transformer[1], "fit"):
                transformer = (transformer[0], TransformerAdapter(transformer[1]))

            adapted_transformer_list.append(transformer)

        super().__init__(
            adapted_transformer_list,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out,
        )

    def __sklearn_clone__(self):
        return AdapterFeatureUnion(
            self._unadapted_transformer_list,
            n_jobs=self.n_jobs,
            transformer_weights=self.transformer_weights,
            verbose=self.verbose,
            verbose_feature_names_out=self.verbose_feature_names_out,
        )

    def transform_eval(self, X, y=None):
        Xs = []

        # TODO: make parallel
        for _, transform, _ in self._iter():
            # TODO: take weight into account
            if hasattr(transform, "transform_eval"):
                X_ = transform.transform_eval(X, y)
            else:
                X_ = transform.transform(X)

            if isinstance(X_, tuple):
                X_, _ = X_

            Xs.append(X_)

        Xt = self._hstack(Xs)
        if y is None:
            return Xt

        return Xt, y


class EvaluatedModel(BaseEstimator, TransformerMixin):
    """Model with evaluation.

    Wraps a model to log info.

    Parameters
    ----------
    model : sklearn.BaseEstimator
        Estimator being evaluated.
    evaluator : polpo.ModelEvaluator
        Model evaluator.
    """

    # TODO: need to think about inheritance
    # TODO: split in two?

    def __init__(self, model, evaluator):
        super().__init__()
        self.model = model
        self.evaluator = evaluator
        self.eval_result_ = None
        self.eval_result_pred_ = None

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model."""
        return getattr(self.model, name)

    def __sklearn_clone__(self):
        return EvaluatedModel(model=clone(self.model), evaluator=self.evaluator)

    def fit(self, X, y=None):
        self.model = self.model.fit(X, y)

        self.eval_result_ = self.evaluator(self.model, X, y)
        return self

    def predict_eval(self, X, y=None):
        check_is_fitted(self)

        if hasattr(self.model, "predict_eval"):
            y_pred = self.model.predict_eval(X, y)
        else:
            y_pred = self.model.predict(X)

        self.eval_result_pred_ = self.evaluator(self.model, X, y, y_pred=y_pred)
        return y_pred

    def transform_eval(self, X, y=None):
        check_is_fitted(self)

        Xt = self.model.transform(X)

        self.eval_result_pred_ = self.evaluator(self.model, X, y)

        return Xt


class SupervisedEmbeddingRegressor(BaseEstimator, RegressorMixin):
    # TODO: how shaky is this?

    def __init__(self, encoder, regressor):
        self.encoder = encoder
        self.regressor = regressor

        self.encoder_ = None
        self.regressor_ = None

    def fit(self, X, y):
        self.encoder_ = clone(self.encoder)
        self.regressor_ = clone(self.regressor)

        self.encoder_.fit(y, X)

        z = self.encoder_.transform(y)
        self.regressor_.fit(X, z)
        return self

    def predict(self, X):
        check_is_fitted(self)

        z_pred = self.regressor_.predict(X)
        return self.encoder_.inverse_transform(z_pred)

    def predict_eval(self, X, y):
        check_is_fitted(self)

        if hasattr(self.encoder_, "transform_eval"):
            yt = self.encoder_.transform_eval(y)
        else:
            yt = self.encoder_.transform(y)

        if hasattr(self.regressor_, "predict_eval"):
            z_pred = self.regressor_.predict_eval(X, yt)
        else:
            z_pred = self.regressor_.predict(X)

        return self.encoder_.inverse_transform(z_pred)


class PValuesAdjuster:
    # TODO: implement own version to avoid installing multipletests?
    # NB: some of them are trivial
    def __init__(self, method="bonferroni"):
        # TODO: add method validation; add also two-stage?
        self.method = method

    def __call__(self, pvals):
        return multipletests(pvals.flatten(), method=self.method)[1].reshape(
            pvals.shape
        )


class OlsPValues(ModelEvaluator):
    def __init__(self, adjust_pvalues=None, prefix="", extender=None):
        super().__init__(prefix, extender)
        if adjust_pvalues is None:
            adjust_pvalues = PValuesAdjuster()

        self.adjust_pvalues = adjust_pvalues

    def __call__(self, model, X, y, y_pred=None):
        # assumes X does not contain intercept

        if y_pred is None:
            y_pred = model.predict(X)

        y = y[:, None] if y.ndim == 1 else y
        y_pred = y_pred[:, None] if y_pred.ndim == 1 else y_pred

        mse = np.mean(((y_pred - y) ** 2), axis=0)

        n = y.shape[0]
        p = X.shape[1] + model.fit_intercept

        res_var = mse * n / (n - p)

        X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]

        cov = np.linalg.inv(X_with_intercept.transpose() @ X_with_intercept)

        # take into account fit_intercept
        index = int(model.fit_intercept)
        cov_beta = res_var[:, None] * (np.diag(cov)[index:])

        std_err = np.sqrt(cov_beta)

        t = model.coef_ / std_err

        df = n - p

        pvalues = 2 * (1 - scipy.stats.t.cdf(abs(t), df))

        res = {
            "mse": mse,
            "res_var": res_var,
            "std_err": std_err,
            "t": t,
            "pvals": pvalues,
        }

        if self.adjust_pvalues is not None:
            res["adj-pvals"] = self.adjust_pvalues(res["pvals"])

        return self._update_results(res)


class RegressionMetricAdapter(ModelEvaluator):
    # adapts sklearn metric
    def __init__(self, name, func, multioutput="raw_values", prefix="", extender=None):
        super().__init__(prefix, extender)
        self.name = name
        self.func = func
        self.multioutput = multioutput

    def __call__(self, model, X, y, y_pred=None):
        # model and X are ignored if y_pred is not None
        if y_pred is None:
            y_pred = model.predict(X)

        return self._update_results(
            {self.name: self.func(y, y_pred, multioutput=self.multioutput)}
        )


class R2Score(RegressionMetricAdapter):
    def __init__(self, multioutput="raw_values", prefix="", extender=None):
        super().__init__(
            "r2", r2_score, multioutput=multioutput, prefix=prefix, extender=extender
        )


class PcaEvaluator(ModelEvaluator):
    def __call__(self, model, X, y=None, y_pred=None):
        return self._update_results(
            {
                "expl_var": model.explained_variance_,
                "expl_var_ratio": model.explained_variance_ratio_,
                "expl_var_ratio-cum": np.cumsum(model.explained_variance_ratio_),
            }
        )


class ReconstructionError(ModelEvaluator):
    def __call__(self, model, X, y=None, y_pred=None):
        # NB: X is flattened at this point
        X_recon = model.inverse_transform(model.transform(X))

        diff2 = (X - X_recon) ** 2

        featurewise_rec_error = np.sum(diff2, axis=0)
        rec_error_sum = np.sum(diff2)
        rec_error_mse = np.mean(diff2)

        return self._update_results(
            {
                "featurewise_rec_error": featurewise_rec_error,
                "rec_error_sum": rec_error_sum,
                "rec_error_mse": rec_error_mse,
            }
        )


class VertexReconstructionError(ModelEvaluator):
    def __init__(self, dim=3, prefix=""):
        super().__init__(prefix)
        self.dim = dim

    def __call__(self, model, X, y=None, y_pred=None):
        # NB: X is flattened at this point
        X_recon = model.inverse_transform(model.transform(X))

        X = X.reshape(X.shape[0], -1, self.dim)
        X_recon = X_recon.reshape(X.shape[0], -1, self.dim)

        diff2 = np.linalg.norm(X - X_recon, axis=-1) ** 2

        vertexwise_rec_error = np.sum(diff2, axis=0)
        rec_error_sum = np.sum(diff2)
        rec_error_mse = np.mean(diff2)

        return self._update_results(
            {
                "vertexwise_rec_error": vertexwise_rec_error,
                "rec_error_sum": rec_error_sum,
                "rec_error_mse": rec_error_mse,
            }
        )


class ShapeCollector(ModelEvaluator):
    # NB: for debug purposes

    def __call__(self, model, X, y=None, y_pred=None):
        return {
            f"{var_name}-shape": var.shape
            for var_name, var in [("X", X), ("y", y), ("y_pred", y_pred)]
            if var is not None
        }


class ResultsExtender:
    def __init__(self, metrics=None):
        self.metrics = metrics

    def __call__(self, results):
        metrics = list(results.keys()) if self.metrics is None else self.metrics

        for metric in metrics:
            val = results[metric]
            results.update(
                {
                    f"{metric}-mean": np.mean(val),
                    f"{metric}-max": np.max(val),
                    f"{metric}-min": np.min(val),
                }
            )

        return results


def collect_eval_results(estimator, unnest=True, outer_key="", train=True):
    # outer_key only applicable if unnest is True
    results = {}

    if isinstance(estimator, EvaluatedModel):
        results["res"] = (
            estimator.eval_result_ if train else estimator.eval_result_pred_
        )

    # Pipeline
    if hasattr(estimator, "named_steps"):
        for name, step in estimator.named_steps.items():
            res = collect_eval_results(step, unnest=False, train=train)
            if len(res):
                results[name] = res

    # FeatureUnion
    if hasattr(estimator, "named_transformers"):
        for name, transformer in estimator.named_transformers.items():
            res = collect_eval_results(transformer, unnest=False, train=train)
            if len(res):
                results[name] = res

    # TransformedTargetRegressor, SupervisedEmbeddingRegressor, not pipeline
    if hasattr(estimator, "regressor_") and not hasattr(estimator, "named_steps"):
        results["regr"] = collect_eval_results(
            estimator.regressor_, unnest=False, train=train
        )

    if hasattr(estimator, "transformer_") and not hasattr(estimator, "named_steps"):
        results["transformer"] = collect_eval_results(
            estimator.transformer_, unnest=False, train=train
        )

    # SupervisedEmbeddingRegressor, not pipeline
    if hasattr(estimator, "encoder_") and not hasattr(estimator, "named_steps"):
        results["encoder"] = collect_eval_results(
            estimator.encoder_, unnest=False, train=train
        )

    if unnest:
        results = _unnest_results(results, current_key=outer_key)

    return results


def _unnest_results(results, current_key="", res_dict=None):
    # TODO: adapt to use unnest_dict?
    sep = "/" if current_key else ""
    if res_dict is None:
        res_dict = {}

    for key, value in results.items():
        if key == "res":
            res_dict[current_key] = value

        else:
            res_dict = _unnest_results(
                value, current_key=f"{current_key}{sep}{key}", res_dict=res_dict
            )

    return res_dict
