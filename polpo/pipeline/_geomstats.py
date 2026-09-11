import geomstats.backend as gs
import numpy as np

from polpo.mesh.surfaces import PvSurface, Surface, TrimeshSurface
from polpo.preprocessing.base import Pipeline, PreprocessingStep

try:
    from ._pyvista import DataFromPv, PvFromData
except ImportError:
    pass


class SurfaceFromData:
    def __call__(self, mesh):
        # TODO: handle colors and signal?
        print(mesh)
        if len(mesh) == 3:
            vertices, faces, _ = mesh
        else:
            vertices, faces = mesh

        return Surface(gs.asarray(vertices), gs.asarray(faces))


class DataFromSurface:
    def __call__(self, mesh):
        # TODO: handle colors and signal?
        return np.asarray(mesh.vertices), np.asarray(mesh.faces)


class SurfaceFromPvMesh(Pipeline):
    def __init__(self):
        super().__init__(steps=[DataFromPv(), SurfaceFromData()])


class PvMeshFromSurface(Pipeline):
    def __init__(self):
        super().__init__(steps=[DataFromSurface(), PvFromData()])


class SurfaceFromTrimesh(PreprocessingStep):
    def __call__(self, mesh):
        return Surface(
            gs.from_numpy(mesh.vertices),
            gs.from_numpy(mesh.faces.astype(np.int64)),
        )


class _SurfaceFromMesh(PreprocessingStep):
    def __init__(self, Surface, clone=False):
        super().__init__()
        self.clone = clone
        self.Surface = Surface

    def __call__(self, mesh):
        if self.clone:
            mesh = mesh.copy()

        return self.Surface(mesh)


class TrimeshSurfaceFromTrimesh(_SurfaceFromMesh):
    def __init__(self, clone=False):
        super().__init__(TrimeshSurface, clone=clone)


class PvSurfaceFromPvMesh(_SurfaceFromMesh):
    def __init__(self, clone=False):
        super().__init__(PvSurface, clone=clone)
