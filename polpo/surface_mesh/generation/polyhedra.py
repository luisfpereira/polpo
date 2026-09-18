import geomstats.backend as gs


def create_regular_tetrahedron(edge_length=1.0):
    """Create the vertices and faces of a regular tetrahedron.

    Parameters
    ----------
    edge_length : float
        Length of each edge.

    Returns
    -------
    vertices : array-like, shape=(4, 3)
        Vertex coordinates.
    faces : array-like, shape=(4, 3)
        Indices of the vertices defining each triangular face.
    """
    scale = edge_length / (2.0 * gs.sqrt(2.0))

    vertices = scale * gs.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )

    faces = gs.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ]
    )

    return vertices, faces
