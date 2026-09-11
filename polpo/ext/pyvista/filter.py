import numpy as np

from polpo.preprocessing.base import PreprocessingStep
from polpo.utils import params_to_kwargs


class PvExtractPoints(PreprocessingStep):
    """Get a subset of the grid (with cells).

    Cells are added if contain any of the given point indices.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.extract_points
    """

    def __init__(
        self,
        adjacent_cells=True,
        include_cells=True,
        progress_bar=False,
        extract_surface=True,
    ):
        super().__init__()
        self.adjacent_cells = adjacent_cells
        self.include_cells = include_cells
        self.progress_bar = progress_bar
        self.extract_surface = extract_surface

    def __call__(self, data):
        """Apply step.

        Parameters
        ----------
        data : tuple[pv.PolyData, indices]
        """
        mesh, indices = data
        subset = mesh.extract_points(
            indices, **params_to_kwargs(self, func=mesh.extract_points)
        )
        if self.extract_surface:
            return subset.extract_surface(algorithm="dataset_surface")

        return subset


class PvSelectSubset(PreprocessingStep):
    """Get subset of a mesh with given value."""

    def __init__(
        self,
        value=None,
        array_name="labels",
        progress_bar=False,
    ):
        super().__init__()
        # assumes int
        self.value = value
        self.array_name = array_name
        self._point_extractor = PvExtractPoints(
            adjacent_cells=True,
            include_cells=True,
            progress_bar=progress_bar,
            extract_surface=True,
        )

    def __call__(self, data):
        if isinstance(data, (list, tuple)):
            mesh, value = data
        else:
            mesh = data
            value = self.value

        ind = np.argwhere(mesh.get_array(self.array_name) == value)

        mesh = self._point_extractor((mesh, ind))

        return mesh


class PvSubsetSplitter(PreprocessingStep):
    def __init__(self, array_name="labels", progress_bar=False):
        super().__init__()
        self._selector = PvSelectSubset(
            array_name=array_name, progress_bar=progress_bar
        )

    def __call__(self, mesh):
        values = np.unique(mesh.get_array(self._selector.array_name))
        return {int(value): self._selector((mesh, value)) for value in values}
