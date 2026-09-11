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


class ObjectBasedTransformedTargetRegressor(SkTransformedTargetRegressor):
    # TODO: delete?
    def fit(self, X, y, **fit_params):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        **fit_params : dict
            - If `enable_metadata_routing=False` (default): Parameters directly passed
              to the `fit` method of the underlying regressor.

            - If `enable_metadata_routing=True`: Parameters safely routed to the `fit`
              method of the underlying regressor.

            .. versionchanged:: 1.6
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if y is None:
            raise ValueError(
                f"This {self.__class__.__name__} estimator "
                "requires y to be passed, but the target y is None."
            )
        self._fit_transformer(y)

        # transform y and convert back to 1d array if needed
        y_trans = self.transformer_.transform(y)

        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)

        self.regressor_ = self._get_regressor(get_clone=True)

        self.regressor_.fit(X, y_trans, **fit_params)

        if hasattr(self.regressor_, "feature_names_in_"):
            self.feature_names_in_ = self.regressor_.feature_names_in_

        return self

    def predict(self, X, **predict_params):
        """Predict using the base regressor, applying inverse.

        The regressor is used to predict and the `inverse_func` or
        `inverse_transform` is applied before returning the prediction.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        **predict_params : dict of str -> object
            Parameters passed to the `predict` method of the underlying
            regressor.

        Returns
        -------
        y_hat : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self)
        pred = self.regressor_.predict(X, **predict_params)
        if pred.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(pred.reshape(-1, 1))
        else:
            pred_trans = self.transformer_.inverse_transform(pred)

        return pred_trans

    def predict_eval(self, X, y):
        check_is_fitted(self)

        if hasattr(self.transformer_, "transform_eval"):
            yt = self.transformer_.transform_eval(y)
        else:
            yt = self.transformer_.transform(y)

        if hasattr(self.regressor_, "predict_eval"):
            pred = self.regressor_.predict_eval(X, yt)
        else:
            pred = self.regressor_.predict(X)

        if pred.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(pred.reshape(-1, 1))
        else:
            pred_trans = self.transformer_.inverse_transform(pred)

        return pred_trans


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
