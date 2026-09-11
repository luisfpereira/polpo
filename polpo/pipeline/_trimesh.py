import os

import numpy as np
import trimesh

from polpo.preprocessing.base import PreprocessingStep
from polpo.preprocessing.mesh._register import register_vertices_attr

try:
    from ._pyvista import PvReader
except ImportError:
    pass


register_vertices_attr(trimesh.Trimesh, "vertices")


class TrimeshFromData(PreprocessingStep):
    def __call__(self, mesh):
        if len(mesh) == 3:
            vertices, faces, colors = mesh
        else:
            vertices, faces = mesh
            colors = None
        return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)


class DataFromTrimesh(PreprocessingStep):
    def __call__(self, mesh):
        return (
            np.array(mesh.vertices),
            np.array(mesh.faces),
            np.array(mesh.visual.vertex_colors),
        )


class TrimeshFromPvMesh(PreprocessingStep):
    def __call__(self, poly_data):
        vertex_colors = (
            poly_data["colors"] if "colors" in poly_data.array_names else None
        )
        faces_as_array = poly_data.faces.reshape((poly_data.n_faces_strict, 4))[:, 1:]
        return trimesh.Trimesh(
            poly_data.points, faces_as_array, vertex_colors=vertex_colors
        )


class TrimeshFaceRemoverByArea(PreprocessingStep):
    # TODO: generalize?

    def __init__(self, threshold=0.01, inplace=True):
        super().__init__()
        self.threshold = threshold
        self.inplace = inplace

    def __call__(self, mesh):
        if not self.inplace:
            mesh = mesh.copy()

        face_mask = ~np.less(mesh.area_faces, self.threshold)
        mesh.update_faces(face_mask)

        return mesh


class TrimeshDegenerateFacesRemover(PreprocessingStep):
    """Trimesh degenerate faces remover.

    https://trimesh.org/trimesh.base.html#trimesh.base.Trimesh.nondegenerate_faces

    Parameters
    ----------
    height: float
        Identifies faces with an oriented bounding box shorter than
        this on one side.
    """

    def __init__(self, height=1e-08, inplace=True):
        super().__init__()
        self.height = height
        self.inplace = inplace

    def __call__(self, mesh):
        if not self.inplace:
            mesh = mesh.copy()

        faces = mesh.nondegenerate_faces(height=self.height)
        mesh.update_faces(faces)
        return mesh


class TrimeshDecimator(PreprocessingStep):
    """Trimesh simplify quadratic decimation.

    Parameters
    ----------
    percent : float
        A number between 0.0 and 1.0 for how much.
    face_count : int
        Target number of faces desired in the resulting mesh.
    agression: int
        An integer between 0 and 10, the scale being roughly 0 is
        “slow and good” and 10 being “fast and bad.”
    """

    # NB: uses fast-simplification

    def __init__(self, percent=None, face_count=None, agression=None):
        super().__init__()
        self.percent = percent
        self.face_count = face_count
        self.agression = agression

    def __call__(self, mesh):
        # TODO: issue with colors?
        decimated_mesh = mesh.simplify_quadric_decimation(
            percent=self.percent,
            face_count=self.face_count,
            aggression=self.agression,
        )

        # # TODO: delete
        # colors_ = np.array(decimated_mesh.visual.vertex_colors)
        # print("unique colors after decimation:", len(np.unique(colors_, axis=0)))

        return decimated_mesh


class TrimeshLargestComponentSelector(PreprocessingStep):
    def __init__(self, only_watertight=False):
        super().__init__()
        self.only_watertight = only_watertight

    def __call__(self, mesh):
        components = mesh.split(only_watertight=self.only_watertight)
        if len(components) == 0:
            return mesh

        components.sort(key=lambda component: len(component.faces), reverse=True)
        return components[0]


class TrimeshToPly(PreprocessingStep):
    def __init__(
        self,
        dirname="",
        encoding="binary",
        vertex_normal=None,
        include_attributes=True,
    ):
        super().__init__()
        # TODO: create dir if does not exist?
        self.dirname = dirname
        self.encoding = encoding
        self.vertex_normal = vertex_normal
        self.include_attributes = include_attributes

        # TODO: add override?

    def __call__(self, data):
        filename, mesh = data

        ext = ".ply"
        if not filename.endswith(ext):
            filename += ext

        path = os.path.join(self.dirname, filename)

        ply_text = trimesh.exchange.ply.export_ply(
            mesh, encoding=self.encoding, include_attributes=self.include_attributes
        )

        with open(path, "wb") as file:
            file.write(ply_text)

        return path


class TrimeshReader(PreprocessingStep):
    """Read file.

    Uses `load_mesh` (
    https://trimesh.org/trimesh.exchange.load.html#trimesh.exchange.load.load_mesh
    ) if supported format, which is very limited.

    If not supported format, goes through `pyvista`.
    Particularly relevant for `vtk` `Polydata`,
    otherwise `meshio` could have been used.
    """

    def __init__(self):
        super().__init__()
        self._supported_fmts = {"stl", "ply", "dxf"}

    def __call__(self, path):
        ext = path.split(".")[-1]
        if ext in self._supported_fmts:
            return trimesh.load_mesh(path)

        return (PvReader() + TrimeshFromPvMesh())(path)


class TrimeshLaplacianSmoothing(PreprocessingStep):
    """
    https://trimesh.org/trimesh.smoothing.html#trimesh.smoothing.filter_laplacian
    """

    def __init__(
        self,
        lamb=0.5,
        iterations=10,
        implicit_time_integration=False,
        volume_constraint=True,
        laplacian_operator=None,
        inplace=True,
    ):
        super().__init__()
        self.lamb = lamb
        self.iterations = iterations
        self.implicit_time_integration = implicit_time_integration
        self.volume_constraint = volume_constraint
        self.laplacian_operator = laplacian_operator
        self.inplace = inplace

    def __call__(self, mesh):
        if not self.inplace:
            mesh = mesh.copy()

        trimesh.smoothing.filter_laplacian(
            mesh,
            lamb=self.lamb,
            iterations=self.iterations,
            implicit_time_integration=self.implicit_time_integration,
            volume_constraint=self.volume_constraint,
            laplacian_operator=self.laplacian_operator,
        )

        return mesh


class TrimeshClone(PreprocessingStep):
    def __call__(self, mesh):
        return mesh.copy()


class TrimeshMeshBounds(PreprocessingStep):
    def __init__(self, ratio=0.0):
        super().__init__()
        self.ratio = ratio

    def __call__(self, mesh):
        bounds = mesh.bounds

        if abs(self.ratio) < 1e-4:
            return bounds

        return np.stack([(1 - self.ratio) * bounds[0], (1 + self.ratio) * bounds[1]])
