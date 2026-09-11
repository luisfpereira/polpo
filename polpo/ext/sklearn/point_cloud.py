from sklearn.preprocessing import FunctionTransformer

from polpo.preprocessing.point_cloud.smoothing import RegisteredPointCloudSmoothing

from .base import GetParamsMixin


class FittableRegisteredPointCloudSmoothing(GetParamsMixin, FunctionTransformer):
    def __init__(self, index=0, n_neighbors=10, smoothing_func=None):
        # index: which mesh to select
        self.index = index

        inverse_func = RegisteredPointCloudSmoothing(
            n_neighbors=n_neighbors, smoothing_func=smoothing_func
        )

        super().__init__(
            inverse_func=inverse_func,
            check_inverse=False,
        )

    def fit(self, X, y=None):
        # y is ignored
        self.inverse_func.fit(X[self.index])
        return self
