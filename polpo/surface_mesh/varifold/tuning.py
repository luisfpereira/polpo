import geomstats.backend as gs

from polpo.surface_mesh.scale import (
    median_edge_length,
    vertex_centroid_to_farthest_vertex,
)


class SigmaFromScale:
    """Compute a varifold kernel scale from surface geometry.

    The scale is determined from both an object-scale estimate and a
    discretization-scale estimate. For each surface, the two estimates are
    rescaled by their respective ratios and the larger value is retained.
    The final sigma is the maximum of these values across all surfaces.

    Parameters
    ----------
    object_scale : callable
        Function mapping a surface to a scalar object-scale estimate.
    discretization_scale : callable
        Function mapping a surface to a scalar discretization-scale estimate.
    object_ratio : float
        Multiplicative factor applied to the object-scale estimate.
    discretization_ratio : float
        Multiplicative factor applied to the discretization-scale estimate.
    """

    def __init__(
        self,
        object_ratio=0.25,
        discretization_ratio=2.0,
        object_scale=None,
        discretization_scale=None,
    ):
        if object_scale is None:
            object_scale = vertex_centroid_to_farthest_vertex

        if discretization_scale is None:
            discretization_scale = median_edge_length

        self.object_ratio = object_ratio
        self.discretization_ratio = discretization_ratio
        self.object_scale = object_scale
        self.discretization_scale = discretization_scale

    def __call__(self, surfaces):
        """Compute sigma from a collection of surfaces.

        Parameters
        ----------
        surfaces : iterable
            Surface meshes used to estimate the kernel scale.

        Returns
        -------
        sigma : scalar
            Varifold kernel scale.
        """
        object_scales = gs.array([self.object_scale(surface) for surface in surfaces])
        discretization_scales = gs.array(
            [self.discretization_scale(surface) for surface in surfaces]
        )

        sigmas = gs.maximum(
            self.object_ratio * object_scales,
            self.discretization_ratio * discretization_scales,
        )

        return gs.amax(sigmas)
