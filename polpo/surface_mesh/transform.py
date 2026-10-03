import geomstats.backend as gs
from geomstats.metric_geometry.vectorization import _manipulate_output, vectorize_point

from polpo.surface_mesh.core import Surface
from polpo.transform import InvertibleTransform


def _output_as_array(out, to_list):
    return _manipulate_output(out, to_list, manipulate_output_iterable=gs.array)


@vectorize_point((0, "point"), manipulate_output=_output_as_array)
def mesh_to_vertices(point):
    return [point_.vertices for point_ in point]


class DeltaTransform(InvertibleTransform):
    """Transform meshes to and from flattened template-relative deltas.

    Parameters
    ----------
    template : Surface
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
        meshes : Surface or array-like of Surface, shape [...]
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
        meshes : Surface or ndarray of Surface, shape [...]
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
            meshes[i] = Surface(
                gs.to_numpy(self.template.vertices + delta), self.template.faces
            )

        if single:
            return meshes[0]

        return meshes.reshape(batch_shape)
