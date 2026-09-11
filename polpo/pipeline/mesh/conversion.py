"""Mesh type conversion."""

import copy
import sys

try:
    from polpo.preprocessing._trimesh import (  # noqa:F401
        DataFromTrimesh,
        TrimeshFromData,
        TrimeshFromPvMesh,
    )
except ImportError:
    pass

try:
    from polpo.preprocessing._pyvista import (  # noqa:F401
        DataFromPv,
        PvFromData,
        PvFromTrimesh,
    )
except ImportError:
    pass

try:
    from polpo.preprocessing._skshapes import PvFromSks, SksFromPv  # noqa:F401
except ImportError:
    pass

try:
    from polpo.preprocessing._geomstats import (  # noqa:F401
        DataFromSurface,
        PvMeshFromSurface,
        PvSurfaceFromPvMesh,
        SurfaceFromData,
        SurfaceFromPvMesh,
        SurfaceFromTrimesh,
        TrimeshSurfaceFromTrimesh,
    )
except ImportError:
    pass

try:
    from polpo.preprocessing._meshio import MeshioFromData  # noqa:F401
except ImportError:
    pass

from polpo.macro import create_to_classes_from_from
from polpo.preprocessing.base import PreprocessingStep

from ._register import VERTICES_ATTR


class ToVertices(PreprocessingStep):
    """Get mesh vertices."""

    def __call__(self, mesh):
        """Apply step.

        Parameters
        ----------
        mesh : Mesh
            Mesh with vertex information.

        Returns
        -------
        vertices : array-like
            Mesh vertices.
        """
        attr = VERTICES_ATTR.get(type(mesh), "vertices")
        return getattr(mesh, attr)


class ToFaces(PreprocessingStep):
    """Get mesh faces."""

    def __call__(self, mesh):
        """Apply step.

        Parameters
        ----------
        mesh : Mesh
            Mesh with face information.

        Returns
        -------
        faces : array-like
            Mesh faces.
        """
        return mesh.faces


class FromCombinatorialStructure(PreprocessingStep):
    """Get mesh from combinatorial structure.

    Parameters
    ----------
    mesh : Mesh
        Mesh containing combinatorial structure.
        If None, must be supplied at apply.
    """

    def __init__(self, mesh=None):
        super().__init__()
        self.mesh = mesh

    def __call__(self, data):
        """Apply step.

        Returns
        -------
        mesh : Mesh
            Mesh with new vertices but same combinatorial structure.
        """
        if isinstance(data, (list, tuple)):
            mesh, vertices = data
        else:
            mesh = self.mesh
            vertices = data

        mesh = copy.copy(mesh)

        vertices_attr = VERTICES_ATTR.get(type(mesh), "vertices")
        setattr(mesh, vertices_attr, vertices)

        return mesh


create_to_classes_from_from(sys.modules[__name__])
