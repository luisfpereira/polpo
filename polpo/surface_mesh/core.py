import geomstats.backend as gs

from polpo.surface_mesh.face import compute_face_info

# TODO: add enclosing ball algorithm and wrap geomstats surface?

# TODO: add tests checking results are the same for all representations


class Surface:
    """A surface mesh.

    Mesh info (face centroids, normals and areas) is cached
    for greater performance.

    Parameters
    ----------
    vertices : array-like, shape=[n_vertices, 3]
        Mesh vertices.
    faces : array-like, shape=[n_faces, 3]
        Mesh faces.

    Attributes
    ----------
    vertices : array-like, shape=[..., n_vertices, 3]
        Mesh vertex coordinates.
    faces : array-like, shape=[n_faces, 3]
        Indices of the three vertices defining each triangular face.
    face_centroids : array-like, shape=[..., n_faces, 3]
        Coordinates of the face centroids.
    face_normals : array-like, shape=[..., n_faces, 3]
        Unit face normals. Their orientation is determined by the vertex
        ordering in ``faces``.
    face_areas : array-like, shape=[..., n_faces]
        Face areas.
    """

    # TODO: add miniball to compute bounding sphere?

    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        (
            self.face_centroids,
            self.face_normals,
            self.face_areas,
        ) = compute_face_info(vertices, faces)

    @property
    def bounds(self):
        return gs.stack(
            (gs.amin(self.vertices, axis=0), gs.amax(self.vertices, axis=0))
        )

    @property
    def vertex_centroid(self):
        # TODO: review
        return gs.mean(self.vertices, axis=0)
