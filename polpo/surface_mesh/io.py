"""Input/output utilities for surface meshes."""

import meshio

from polpo.surface_mesh.core import Surface


def _to_native(array):
    dtype = array.dtype.newbyteorder("=")
    return array.astype(dtype, copy=False)


def write_surface_data(path, vertices, faces):
    """Write a triangular surface mesh to disk.

    Parameters
    ----------
    path : path-like
        Output file path.
    vertices : array-like, shape=[n_vertices, 3]
        Mesh vertex coordinates.
    faces : array-like, shape=[n_faces, 3]
        Indices of the three vertices defining each triangular face.
    """
    meshio.write_points_cells(
        path,
        vertices,
        [("triangle", faces)],
    )


def write_surface(path, surface):
    """Write a surface mesh to disk.

    Parameters
    ----------
    path : path-like
        Output file path.
    surface : surface-like
        Surface mesh exposing ``vertices`` and ``faces`` attributes.
    """
    write_surface_data(
        path,
        surface.vertices,
        surface.faces,
    )


def read_surface_data(path):
    """Read surface mesh data from disk.

    Parameters
    ----------
    path : path-like
        Input file path.

    Returns
    -------
    vertices : array-like, shape=[n_vertices, 3]
        Mesh vertex coordinates.
    faces : array-like, shape=[n_faces, 3]
        Indices of the three vertices defining each triangular face.
    """
    mesh = meshio.read(path)

    vertices = _to_native(mesh.points)
    faces = _to_native(mesh.get_cells_type("triangle"))

    return vertices, faces


def read_surface(path):
    """Read a surface mesh from disk.

    Parameters
    ----------
    path : path-like
        Input file path.

    Returns
    -------
    surface : Surface
        Surface mesh.
    """
    return Surface(*read_surface_data(path))
