"""Trimesh surface mesh representation."""

import geomstats.backend as gs
import trimesh

from polpo.surface_mesh.core import VerticesFacesMixin


class TrimeshSurface(VerticesFacesMixin):
    """Surface mesh backed by ``trimesh.Trimesh``.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Triangular surface mesh.

    Attributes
    ----------
    trimesh : trimesh.Trimesh
        Underlying Trimesh surface mesh.
    """

    def __init__(self, mesh):
        super().__init__()
        self.trimesh = mesh

    @classmethod
    def from_data(cls, vertices, faces, process=False):
        """Create a surface from vertices and faces.

        Parameters
        ----------
        vertices : array-like, shape=[n_vertices, 3]
            Mesh vertex coordinates.
        faces : array-like, shape=[n_faces, 3]
            Indices of the three vertices defining each triangular face.
        process : bool
            Whether to process the mesh on construction.

        Returns
        -------
        surface : TrimeshSurface
            Trimesh-backed surface mesh.
        """
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )
        return cls(mesh)

    @property
    def vertices(self):
        """Return the mesh vertices.

        Returns
        -------
        vertices : array-like, shape=[n_vertices, 3]
            Mesh vertex coordinates.
        """
        return gs.asarray(self.trimesh.vertices)

    @property
    def faces(self):
        """Return the mesh faces.

        Returns
        -------
        faces : array-like, shape=[n_faces, 3]
            Indices of the three vertices defining each triangular face.
        """
        return gs.asarray(self.trimesh.faces)

    @property
    def face_centroids(self):
        """Return the face centroids.

        Returns
        -------
        face_centroids : array-like, shape=[n_faces, 3]
            Coordinates of the face centroids.
        """
        return gs.asarray(self.trimesh.triangles_center)

    @property
    def face_areas(self):
        """Return the face areas.

        Returns
        -------
        face_areas : array-like, shape=[n_faces, 1]
            Face areas.
        """
        return gs.expand_dims(
            gs.asarray(self.trimesh.area_faces),
            axis=-1,
        )

    @property
    def face_normals(self):
        """Return the face normals.

        Returns
        -------
        face_normals : array-like, shape=[n_faces, 3]
            Unit face normals.
        """
        return gs.asarray(self.trimesh.face_normals)

    @property
    def surface_centroid(self):
        """Compute the centroid of the triangulated surface.

        Returns
        -------
        centroid : array-like, shape=(3,)
            Area-weighted mean of the face centroids.
        """
        return gs.asarray(self.trimesh.centroid)

    @property
    def edges(self):
        """Return the unique mesh edges.

        Returns
        -------
        edges : array-like, shape=[n_edges, 2]
            Indices of the vertices defining each edge.
        """
        return gs.asarray(self.trimesh.edges_unique)

    @property
    def edge_lengths(self):
        """Return the mesh edge lengths.

        Returns
        -------
        edge_lengths : array-like, shape=[n_edges]
            Euclidean length of each edge.
        """
        return gs.asarray(self.trimesh.edges_unique_length)

    @property
    def bounds(self):
        """Return the axis-aligned bounding box limits.

        Returns
        -------
        bounds : array-like, shape=(2, 3)
            Minimum and maximum coordinates along each spatial axis.
        """
        return gs.asarray(self.trimesh.bounds)
