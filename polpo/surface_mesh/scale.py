"""Scale estimators for triangular surface meshes."""

import geomstats.backend as gs


def vertex_centroid_to_farthest_vertex(surface):
    """Compute the maximum distance from the vertex centroid to a vertex.

    Parameters
    ----------
    surface : surface-like
        Surface mesh.

    Returns
    -------
    distance : float
        Maximum Euclidean distance from the vertex centroid to the mesh
        vertices.
    """
    return gs.amax(
        gs.linalg.norm(surface.vertex_centroid - surface.vertices, axis=-1),
    )


def surface_centroid_to_farthest_vertex(surface):
    """Compute the maximum distance from the surface centroid to a vertex.

    Parameters
    ----------
    surface : surface-like
        Surface mesh.

    Returns
    -------
    distance : float
        Maximum Euclidean distance from the surface centroid to the mesh
        vertices.
    """
    return gs.amax(
        gs.linalg.norm(surface.surface_centroid - surface.vertices, axis=-1),
    )


def median_edge_length(surface):
    """Compute the median edge length of a surface mesh.

    Parameters
    ----------
    surface : surface-like
        Surface mesh.

    Returns
    -------
    edge_length : float
        Median length of the mesh edges.
    """
    return gs.median(surface.edge_lengths)
