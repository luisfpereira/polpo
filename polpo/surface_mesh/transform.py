import geomstats.backend as gs
from geomstats.metric_geometry.vectorization import (
    _manipulate_output as gs_manipulate_output,
)
from geomstats.metric_geometry.vectorization import (
    vectorize_point,
)

from polpo.ext.pyvista.surface_mesh import PvSurface
from polpo.pipeline.mesh.conversion import PvFromData
from polpo.transform import InvertibleTransform


def _output_as_array(out, to_list):
    return gs_manipulate_output(out, to_list, manipulate_output_iterable=gs.array)


@vectorize_point((0, "point"), manipulate_output=_output_as_array)
def mesh_to_vertices(point):
    return [point_.vertices for point_ in point]


class VerticesToPvSurface:
    def __init__(self, faces):
        self.faces = faces
        self._from_data = PvFromData() + PvSurface

    def __call__(self, point):
        if len(point.shape) == 2:
            return self._from_data((point, self.faces))

        return [self._from_data((point_, self.faces)) for point_ in point]


class PvSurfaceToVertices:
    def __init__(self, faces):
        self._array_to_pv_surface = VerticesToPvSurface(faces)

    def __call__(self, base_point):
        return mesh_to_vertices(base_point)

    def inverse(self, image_point):
        return self._array_to_pv_surface(image_point)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        # TODO: need to check vectorization
        return gs.asarray(tangent_vec)

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        # TODO: need to check vectorization
        return gs.asarray(image_tangent_vec)


class DeltaTransform(InvertibleTransform):
    """Transform meshes to and from flattened template-relative deltas.

    Parameters
    ----------
    template : PvSurface
        Reference mesh defining the common vertex topology and origin for the
        displacement representation.

    Notes
    -----
    The forward transform maps each mesh to the flattened vertex displacement

        mesh.vertices - template.vertices

    with shape ``(..., 3 * n_vertices)``.

    The inverse transform reshapes these displacements to vertex coordinates,
    adds them to the template vertices, and returns meshes with the template
    topology.
    """

    def __init__(self, template):
        self.template = template
        self.n_vertices = len(template.vertices)

    def __call__(self, meshes):
        """Transform meshes into flattened vertex deltas.

        Parameters
        ----------
        meshes : PvSurface or array-like of PvSurface, shape [...]
            Mesh or collection of meshes sharing the template topology.

        Returns
        -------
        deltas : array-like, shape [..., 3 * n_vertices]
            Flattened vertex displacements relative to the template.
        """
        meshes = gs.asarray(meshes, dtype=object)
        batch_shape = meshes.shape

        deltas = [
            gs.reshape(mesh.vertices - self.template.vertices, (-1,))
            for mesh in meshes.flat
        ]

        return gs.reshape(
            gs.stack(deltas),
            batch_shape + (3 * self.n_vertices,),
        )

    def inverse(self, deltas):
        """Transform flattened vertex deltas back into meshes.

        Parameters
        ----------
        deltas : array-like, shape [..., 3 * n_vertices]
            Flattened vertex displacements relative to the template.

        Returns
        -------
        meshes : PvSurface or ndarray of PvSurface, shape [...]
            Reconstructed mesh or collection of meshes with the template
            topology.
        """
        single = deltas.ndim == 1

        if single:
            deltas = deltas[None, ...]

        batch_shape = deltas.shape[:-1]
        deltas = gs.reshape(deltas, (-1, self.n_vertices, 3))

        meshes = gs.empty(len(deltas), dtype=object)

        for i, delta in enumerate(deltas):
            pv_mesh = self.template.as_pv().copy(deep=True)
            pv_mesh.points = gs.to_numpy(self.template.vertices + delta)
            meshes[i] = PvSurface(pv_mesh)

        if single:
            return meshes[0]

        return meshes.reshape(batch_shape)
