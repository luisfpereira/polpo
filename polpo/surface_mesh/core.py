"""Surface mesh representation."""

import geomstats.backend as gs

from polpo.surface_mesh.face import compute_face_info


class VerticesFacesMixin:
    """Mixin for triangular meshes exposing vertices and faces.

    Subclasses must provide ``vertices`` and ``faces`` attributes.
    """

    @property
    def vertex_centroid(self):
        """Return the centroid of the mesh vertices.

        Returns
        -------
        centroid : array-like, shape=(3,)
            Arithmetic mean of the vertex coordinates.
        """
        return gs.mean(self.vertices, axis=0)

    @property
    def surface_centroid(self):
        """Return the centroid of the triangulated surface.

        Returns
        -------
        centroid : array-like, shape=(3,)
            Area-weighted mean of the face centroids.
        """
        weighted_centroids = self.face_areas * self.face_centroids
        return gs.sum(weighted_centroids, axis=0) / gs.sum(self.face_areas)

    @property
    def edges(self):
        """Return the unique edges of the mesh.

        Returns
        -------
        edges : array-like, shape=(n_edges, 2)
            Indices of the vertices defining each edge.
        """
        vind012 = gs.concatenate([self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]])
        vind120 = gs.concatenate([self.faces[:, 1], self.faces[:, 2], self.faces[:, 0]])
        edges = gs.stack(
            [
                gs.concatenate([vind012, vind120]),
                gs.concatenate([vind120, vind012]),
            ],
            axis=-1,
        )
        edges = gs.unique(edges, axis=0)
        return edges[edges[:, 1] > edges[:, 0]]

    @property
    def edge_lengths(self):
        """Return the lengths of the mesh edges.

        Returns
        -------
        edge_lengths : array-like, shape=(n_edges,)
            Euclidean length of each edge.
        """
        edge_points = self.vertices[self.edges]

        return gs.linalg.norm(edge_points[..., 0, :] - edge_points[..., 1, :], axis=-1)

    @property
    def bounds(self):
        """Return the axis-aligned bounding box limits.

        Returns
        -------
        bounds : array-like, shape=(2, 3)
            Minimum and maximum coordinates along each spatial axis.
        """
        return gs.stack(
            (
                gs.amin(self.vertices, axis=0),
                gs.amax(self.vertices, axis=0),
            )
        )


class Surface(VerticesFacesMixin):
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
    face_areas : array-like, shape=[..., n_faces, 1]
        Face areas.
    """

    def __init__(self, vertices, faces):
        super().__init__()

        self.vertices = vertices
        self.faces = faces
        (
            self.face_centroids,
            self.face_normals,
            self.face_areas,
        ) = compute_face_info(vertices, faces)
