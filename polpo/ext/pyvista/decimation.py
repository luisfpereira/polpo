import numpy as np
from sklearn.neighbors import NearestNeighbors

from polpo.preprocessing.base import PreprocessingStep
from polpo.utils import params_to_kwargs


class PvDecimate(PreprocessingStep):
    """Reduce the number of triangles in a triangular mesh.

    Uses vtkQuadricDecimation.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.polydatafilters.decimate#pyvista.PolyDataFilters.decimate
    """

    def __init__(
        self,
        target_reduction,
        volume_preservation=True,
        attribute_error=False,
        scalars=True,
        vectors=True,
        normals=False,
        tcoords=True,
        tensors=True,
        scalars_weight=0.1,
        vectors_weight=0.1,
        normals_weight=0.1,
        tcoords_weight=0.1,
        tensors_weight=0.1,
        inplace=False,
        progress_bar=False,
        keep_colors=True,
    ):
        super().__init__()
        self.target_reduction = target_reduction
        self.volume_preservation = volume_preservation
        self.attribute_error = attribute_error
        self.scalars = scalars
        self.vectors = vectors
        self.normals = normals
        self.tcoords = tcoords
        self.tensors = tensors
        self.scalars_weight = scalars_weight
        self.vectors_weight = vectors_weight
        self.normals_weight = normals_weight
        self.tcoords_weight = tcoords_weight
        self.tensors_weight = tensors_weight
        self.inplace = inplace
        self.progress_bar = progress_bar

        self.keep_colors = keep_colors
        self._nbrs = NearestNeighbors(n_neighbors=1)

    def __call__(self, poly_data):
        """Apply step.

        Parameters
        ----------
        mesh : pv.PolyData
            Mesh to decimate.

        Returns
        -------
        decimated_mesh : pv.PolyData
            Decimated mesh.
        """
        colors = None
        if self.keep_colors and "colors" in poly_data.array_names:
            colors = poly_data["colors"]
            vertices = poly_data.points

        poly_data = poly_data.decimate(
            **params_to_kwargs(self, ignore_private=True, ignore=("keep_colors",))
        )

        if colors is None:
            return poly_data

        self._nbrs.fit(vertices)
        _, indices = self._nbrs.kneighbors(poly_data.points)
        indices = np.squeeze(indices)
        poly_data["colors"] = colors[indices]

        return poly_data
