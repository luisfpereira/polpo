import numpy as np

from polpo.surface_mesh.face import compute_face_areas

# TODO: make gs compatible


def vertex_areas(vertices, faces):
    """Compute barycentric vertex areas.

    Each triangular face contributes one third of its area to each of
    its three incident vertices.

    Parameters
    ----------
    vertices : array-like, shape=[..., n_vertices, 3]
        Mesh vertex coordinates. Leading dimensions are interpreted as
        batch dimensions.
    faces : array-like, shape=[n_faces, 3]
        Indices of the three vertices defining each triangular face.
        The same face topology is shared across all batch dimensions.

    Returns
    -------
    areas : array-like, shape=[..., n_vertices]
        Barycentric area associated with each vertex.
    """
    # TODO: verify and rename?
    face_areas = compute_face_areas(vertices, faces)

    areas = np.zeros(len(vertices), dtype=float)

    np.add.at(areas, faces[:, 0], face_areas / 3.0)
    np.add.at(areas, faces[:, 1], face_areas / 3.0)
    np.add.at(areas, faces[:, 2], face_areas / 3.0)

    return areas
