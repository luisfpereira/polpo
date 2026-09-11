import numpy as np
from sklearn.base import BaseEstimator


class NanEstimator(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.is_fitted_ = True

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.full(X.shape[0], np.nan)
