from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor as SkTransformedTargetRegressor
from sklearn.utils.validation import check_is_fitted


class TransformedTargetRegressor(RegressorMixin, BaseEstimator):
    """Regressor with a transformation applied to the target.

    The target is transformed before fitting the regressor, and predictions
    are mapped back to the original target space using the inverse transform.

    Parameters
    ----------
    regressor : estimator
        Regressor fitted on the transformed target.
    transformer : transformer
        Transformer applied to the target before regression. It must implement
        ``fit``, ``transform``, and ``inverse_transform``.

    Notes
    -----
    This class is analogous to
    :class:`sklearn.compose.TransformedTargetRegressor`, but does not require
    the original target to be a numeric array. This is needed for structured
    targets, such as meshes or other Polpo objects, that are first transformed
    into a numerical representation suitable for an sklearn regressor.
    """

    def __init__(self, regressor, transformer):
        self.regressor = regressor
        self.transformer = transformer

    def fit(self, X, y):
        """Fit the target transformer and regressor.

        Parameters
        ----------
        X : array-like
            Regression covariates.
        y : object
            Targets in the original target space.

        Returns
        -------
        self
            Fitted estimator.
        """
        self.transformer_ = clone(self.transformer)
        self.regressor_ = clone(self.regressor)

        y_transformed = self.transformer_.fit_transform(y)
        self.regressor_.fit(X, y_transformed)

        return self

    def predict(self, X):
        """Predict targets in the original target space.

        Parameters
        ----------
        X : array-like
            Regression covariates.

        Returns
        -------
        y_pred : object
            Predictions mapped back through the inverse target transform.
        """
        y_transformed = self.regressor_.predict(X)
        return self.transformer_.inverse_transform(y_transformed)


class PostTransformingEstimator:
    """Estimator wrapper that learns a post-processing transform after model fitting.

    It applies a learned post-processing transformation to the output of a base estimator.

    The `post_transform` is trained after fitting the base estimator, with access
    to the fitted model, input features `X`, and true targets `y`. This allows
    the transformation to depend on model behavior.

    During `predict`, the base estimator's prediction is passed through the
    trained `post_transform`.

    Parameters
    ----------
    estimator : object
        The base estimator implementing `fit` and `predict`.
    post_transform : object
        A callable with `.fit(model, X, y)` and `__call__(pred)`
        for transforming predictions.
    """

    # TODO: review

    def __init__(self, model, post_transform):
        # TODO: rename to base_model?
        self.model = model
        self.post_transform = post_transform

    def __getattr__(self, name):
        """Get attribute.

        It is only called when ``__getattribute__`` fails.
        Delegates attribute calling to model.
        """
        return getattr(self.model, name)

    def __sklearn_clone__(self):
        # TODO: also clone post_transform
        return PostTransformingEstimator(
            model=clone(self.model), post_transform=self.post_transform
        )

    def fit(self, X, y=None):
        self.model.fit(X, y=y)

        self.post_transform.fit(self.model, X, y)

        return self

    def predict(self, X):
        # NB: X is not transformed yet
        objs = self.model.predict(X)
        return self.post_transform(objs)


class PiecewiseEstimator(BaseEstimator):
    def __init__(self, estimators, partitioner):
        if not isinstance(estimators, (list, tuple)):
            estimators = [estimators] * partitioner.n_bins

        self.estimators = estimators
        self.partitioner = partitioner
        self.estimators_ = None

    def __sklearn_clone__(self):
        return PiecewiseEstimator(
            estimators=[model for model in self.estimators],
            partitioner=self.partitioner,
        )

    def fit(self, X, y=None):
        args = [y] if y is not None else []
        bin_out = self.partitioner(X, *args)

        if y is None:
            bin_X = bin_out
            bin_y = [None] * len(bin_X)

        else:
            bin_X, bin_y = bin_out

        self.estimators_ = []
        for estimator, X_, y_ in zip(self.estimators, bin_X, bin_y):
            self.estimators_.append(clone(estimator).fit(X_, y=y_))

        return self

    def predict(self, X):
        bin_X, indices = self.partitioner(X, recon=True)

        pred = []
        for estimator_, X_ in zip(self.estimators_, bin_X):
            pred.append(estimator_.predict(X_))

        return self.partitioner.reconstruct(indices, pred)
