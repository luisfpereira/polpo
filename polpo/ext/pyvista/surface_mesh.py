"""PyVista surface mesh representation."""

import geomstats.backend as gs
import pyvista as pv

from polpo.surface_mesh.core import VerticesFacesMixin


class PvSurface(VerticesFacesMixin):
    """Surface mesh backed by PyVista ``PolyData``.

    Parameters
    ----------
    polydata : pyvista.PolyData
        Triangular surface mesh.

    Attributes
    ----------
    polydata : pyvista.PolyData
        Underlying PyVista surface mesh.
    """

    def __init__(self, polydata):
        self.polydata = polydata

    @classmethod
    def from_data(cls, vertices, faces):
        """Create a surface from vertices and faces.

        Parameters
        ----------
        vertices : array-like, shape=[n_vertices, 3]
            Mesh vertex coordinates.
        faces : array-like, shape=[n_faces, 3]
            Indices of the three vertices defining each triangular face.

        Returns
        -------
        surface : PvSurface
            PyVista-backed surface mesh.
        """
        polydata = pv.PolyData.from_regular_faces(vertices, faces)
        return cls(polydata)

    @property
    def vertices(self):
        """Return the mesh vertices.

        Returns
        -------
        vertices : array-like, shape=[n_vertices, 3]
            Mesh vertex coordinates.
        """
        return gs.asarray(self.polydata.points)

    @property
    def faces(self):
        """Return the mesh faces.

        Returns
        -------
        faces : array-like, shape=[n_faces, 3]
            Indices of the three vertices defining each triangular face.
        """
        return gs.asarray(self.polydata.regular_faces)

    @property
    def face_areas(self):
        """Return the face areas.

        Returns
        -------
        face_areas : array-like, shape=[n_faces, 1]
            Face areas.
        """
        return gs.expand_dims(
            gs.asarray(self.polydata.compute_cell_sizes()["Area"]),
            axis=-1,
        )

    @property
    def face_centroids(self):
        """Return the face centroids.

        Returns
        -------
        face_centroids : array-like, shape=[n_faces, 3]
            Coordinates of the face centroids.
        """
        return gs.asarray(self.polydata.cell_centers().points)

    @property
    def face_normals(self):
        """Return the face normals.

        Returns
        -------
        face_normals : array-like, shape=[n_faces, 3]
            Unit face normals.
        """
        return gs.asarray(self.polydata.face_normals)

    @property
    def bounds(self):
        """Return the axis-aligned bounding box limits.

        Returns
        -------
        bounds : array-like, shape=(2, 3)
            Minimum and maximum coordinates along each spatial axis.
        """
        return gs.moveaxis(gs.reshape(gs.asarray(self.polydata.bounds), (3, 2)), 0, 1)

    def as_polydata(self):
        """Return the underlying PyVista ``PolyData``."""
        return self.polydata
