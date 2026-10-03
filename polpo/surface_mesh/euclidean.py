import geomstats.backend as gs
from geomstats.geometry.discrete_surfaces import (
    DiscreteSurfaces,
)
from geomstats.geometry.discrete_surfaces import (
    L2SurfacesMetric as _L2SurfacesMetric,
)

from polpo.surface_mesh.geometry import PullbackMetric, SurfacesSpace
from polpo.surface_mesh.transform import mesh_to_vertices


def EuclideanSurfaces(faces):
    # assumes meshes are in correspondence
    image_space = DiscreteSurfaces(faces, equip=False).equip_with_metric(
        L2SurfacesMetric
    )

    return SurfacesSpace().equip_with_metric(
        PullbackMetric,
        forward_map=mesh_to_vertices,
        image_space=image_space,
    )


class L2SurfacesMetric(_L2SurfacesMetric):
    # TODO: implement in geomstats

    def parallel_transport(
        self, tangent_vec, base_point=None, direction=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vector.

        On a Euclidean space, the parallel transport of a (tangent) vector
        returns the vector itself.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., dim]
            Point on the manifold. Point to transport from.
            Optional, default: None
        direction : array-like, shape=[..., dim]
            Tangent vector at base point, along which the parallel transport
            is computed.
            Optional, default: None.
        end_point : array-like, shape=[..., dim]
            Point on the manifold. Point to transport to.
            Optional, default: None.

        Returns
        -------
        transported_tangent_vec : array-like, shape=[..., dim]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.
        """
        # TODO: fix vectorization
        return gs.copy(tangent_vec)
