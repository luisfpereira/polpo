import numpy as np
from sklearn.neighbors import NearestNeighbors

from polpo.preprocessing.base import PreprocessingStep


class RegisteredPointCloudSmoothing(PreprocessingStep):
    def __init__(self, points=None, n_neighbors=10, smoothing_func=None):
        if smoothing_func is None:
            smoothing_func = np.mean

        self.smoothing_func = smoothing_func
        self._nbrs = NearestNeighbors(n_neighbors=n_neighbors)

        self.neighbor_indices = None

        if points is not None:
            self.fit(points)

    def fit(self, points):
        self._nbrs.fit(points)
        self.neighbor_indices = self._nbrs.kneighbors(points, return_distance=False)
        return self.neighbor_indices

    def __call__(self, data):
        # points: (n_samples, n_vertices, dim)
        if self.neighbor_indices is None:
            template_points, points = data
            neighbor_indices = self.fit(template_points)
        else:
            neighbor_indices = self.neighbor_indices
            points = data

        return self.smoothing_func(points[:, neighbor_indices], axis=-2)
