import os
from pathlib import Path

import numpy as np
import pyvista as pv

from polpo.preprocessing.base import PreprocessingStep, RegistrationStep
from polpo.preprocessing.mesh._register import register_vertices_attr
from polpo.pyvista.conversion import DataFromPv, PvFromData  # noqa: F401
from polpo.pyvista.decimation import PvDecimate  # noqa: F401
from polpo.pyvista.filter import (  # noqa: F401
    PvExtractPoints,
    PvSelectSubset,
    PvSubsetSplitter,
)
from polpo.pyvista.io import PvReader  # noqa: F401
from polpo.utils import params_to_kwargs

register_vertices_attr(pv.PolyData, "points")


class PvAlign(RegistrationStep):
    """Align a dataset to another with ICP.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.align
    """

    def __init__(
        self,
        target=None,
        max_landmarks=100,
        max_mean_distance=1e-05,
        max_iterations=500,
        check_mean_distance=True,
        start_by_matching_centroids=True,
        return_matrix=False,
    ):
        super().__init__(target=target)
        self.max_landmarks = max_landmarks
        self.max_mean_distance = max_mean_distance
        self.max_iterations = max_iterations
        self.check_mean_distance = check_mean_distance
        self.start_by_matching_centroids = start_by_matching_centroids
        self.return_matrix = return_matrix

    def __call__(self, data):
        """Apply step.

        Parameters
        ----------
        data : pv.Polydata or tuple[pv.PolyData; 2]
            (source, target) meshes.

        Returns
        -------
        mesh : pv.PolyData
            Source aligned to target.
        matrix : numpy.ndarray
            Transform matrix to transform the input dataset to the target dataset.
        """
        source, target = self._get_source_and_target(data)

        return source.align(
            target,
            **params_to_kwargs(self, ignore=("target",)),
        )


class PvSmoothTaubin(PreprocessingStep):
    """Smooth a PolyData DataSet with Taubin smoothing.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.polydatafilters.smooth_taubin#pyvista.PolyDataFilters.smooth_taubin
    """

    def __init__(
        self,
        n_iter=20,
        pass_band=0.1,
        edge_angle=15.0,
        feature_angle=45.0,
        boundary_smoothing=True,
        feature_smoothing=False,
        non_manifold_smoothing=False,
        normalize_coordinates=False,
        inplace=False,
        progress_bar=False,
    ):
        self.n_iter = n_iter
        self.pass_band = pass_band
        self.edge_angle = edge_angle
        self.feature_angle = feature_angle
        self.boundary_smoothing = boundary_smoothing
        self.feature_smoothing = feature_smoothing
        self.non_manifold_smoothing = non_manifold_smoothing
        self.normalize_coordinates = normalize_coordinates
        self.inplace = inplace
        self.progress_bar = progress_bar

    def __call__(self, poly_data):
        """Apply step."""
        return poly_data.smooth_taubin(**params_to_kwargs(self))


class PvFromTrimesh(PreprocessingStep):
    """Convert trimesh.Trimesh into pv.PolyData."""

    def __init__(self, store_colors=True):
        super().__init__()
        self.store_colors = store_colors

    def __call__(self, mesh):
        """Apply step.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Mesh to convert.

        Returns
        -------
        mesh : pv.PolyData
            Converted mesh.
        """
        pvmesh = pv.wrap(mesh)
        if self.store_colors and hasattr(mesh.visual, "vertex_colors"):
            pvmesh["colors"] = np.array(mesh.visual.vertex_colors)

        return pvmesh


class PvWriter(PreprocessingStep):
    """Write a surface mesh to disk.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata.save
    """

    def __init__(
        self,
        dirname="",
        ext=None,
        binary=True,
        recompute_normals=False,
        exists_ok=True,
    ):
        super().__init__()
        self.dirname = dirname
        self.ext = ext
        self.exists_ok = exists_ok

        self.binary = binary
        self.recompute_normals = recompute_normals

    def __call__(self, data):
        """Apply step.

        Parameters
        ----------
        data : tuple[str, pv.PolyData]
            Filename and mesh to write.

        Returns
        -------
        path : str
            Filename.
        """
        filename, poly_data = data

        if "." not in Path(filename).name and self.ext is not None:
            filename += f".{self.ext}"

        ext = filename.split(".")[1]

        path = Path(os.path.join(self.dirname, filename))
        path.parent.mkdir(parents=True, exist_ok=self.exists_ok)

        texture = (
            poly_data["colors"]
            if ext == "ply" and "colors" in poly_data.array_names
            else None
        )

        poly_data.save(
            path,
            binary=self.binary,
            texture=texture,
            recompute_normals=self.recompute_normals,
        )

        return path


class PvExtractLargest(PreprocessingStep):
    """Extract largest connected set in mesh.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.extract_largest#pyvista.DataSetFilters.extract_largest
    """

    def __init__(self, inplace=False, progress_bar=False):
        super().__init__()
        self.inplace = inplace
        self.progress_bar = progress_bar

    def __call__(self, poly_data):
        """Apply step.

        Parameters
        ----------
        poly_data : pv.PolyData
            Mesh.

        Returns
        -------
        mesh : pv.PolyData
            Largest mesh component.
        """
        return poly_data.extract_largest(**params_to_kwargs(self))
