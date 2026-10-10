from sklearn.base import BaseEstimator, clone


class Reconstructor(BaseEstimator):
    """Reconstruct observations using a fitted transformer.

    Parameters
    ----------
    transformer : object
        Transformer implementing ``fit``, ``transform``, and
        ``inverse_transform``.
    """

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        """Fit a cloned transformer to the observations.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training observations.
        y : array-like or None
            Target values passed to the transformer.

        Returns
        -------
        self : Reconstructor
            Fitted estimator.
        """
        self.transformer_ = clone(self.transformer)
        self.transformer_.fit(X, y)
        return self

    def predict(self, X):
        """Reconstruct observations using the fitted transformer.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Observations to reconstruct.

        Returns
        -------
        X_reconstructed : array-like, shape (n_samples, n_features)
            Reconstructed observations.
        """
        latent = self.transformer_.transform(X)
        return self.transformer_.inverse_transform(latent)
