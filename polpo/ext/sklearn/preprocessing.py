from sklearn.base import BaseEstimator, TransformerMixin


class ColumnIndexSelector(BaseEstimator, TransformerMixin):
    def __init__(self, start_index, end_index):
        super().__init__()
        self.start_index = start_index
        self.end_index = end_index

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.start_index : self.end_index]
