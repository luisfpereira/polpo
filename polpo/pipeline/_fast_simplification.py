import fast_simplification
import numpy as np
from sklearn.neighbors import NearestNeighbors

from polpo.preprocessing.base import PreprocessingStep


class FastSimplificationDecimator(PreprocessingStep):
    def __init__(self, target_reduction=0.25, keep_colors=True):
        super().__init__()
        self.target_reduction = target_reduction
        self.keep_colors = keep_colors

        self._nbrs = NearestNeighbors(n_neighbors=1)

    def __call__(self, mesh):
        if len(mesh) == 3:
            vertices, faces, colors = mesh
        else:
            vertices, faces = mesh
            colors = None

        vertices_, faces_ = fast_simplification.simplify(
            vertices, faces, self.target_reduction
        )

        if not self.keep_colors or colors is None:
            return vertices_, faces_

        self._nbrs.fit(vertices)
        _, indices = self._nbrs.kneighbors(vertices_)
        indices = np.squeeze(indices)
        colors_ = colors[indices]

        return vertices_, faces_, colors_
