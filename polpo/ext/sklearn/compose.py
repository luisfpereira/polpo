from sklearn.base import BaseEstimator, RegressorMixin, clone


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
