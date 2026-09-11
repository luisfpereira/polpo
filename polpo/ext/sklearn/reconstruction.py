from sklearn.base import BaseEstimator, clone


class Reconstructor(BaseEstimator):
    """Use a transformer as a reconstruction estimator."""

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer_ = clone(self.transformer)
        self.transformer_.fit(X, y)
        return self

    def predict(self, X):
        latent = self.transformer_.transform(X)
        return self.transformer_.inverse_transform(latent)
