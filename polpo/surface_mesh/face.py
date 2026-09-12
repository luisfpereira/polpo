"""Utilities for computations on triangular mesh faces.

This module provides basic operations on face geometry, including vertex
coordinates, centroids, oriented area vectors, unit normals, and areas.

Vertex coordinates may have leading batch dimensions, while the face
connectivity is shared across the batch.
"""

import geomstats.backend as gs

from polpo.utils.array import batch_slices


def compute_face_coordinates(vertices, faces):
    """Gather coordinates of vertices defining each face.

    Parameters
    ----------
    vertices : array-like, shape=[..., n_vertices, 3]
        Mesh vertices.
    faces : array-like, shape=[n_faces, 3]
        Triangular mesh faces.

    Returns
    -------
    face_coordinates : array-like, shape=[..., n_faces, 3, 3]
        Coordinates of the three vertices of every face.
    """
    batch_slc = batch_slices(vertices, n_nonbatch_dims=2)
    return vertices[batch_slc + (faces,)]


def compute_face_vertices(vertices, faces):
    """Return the three vertex-coordinate arrays defining each face.

    Parameters
    ----------
    vertices : array-like, shape=[..., n_vertices, 3]
        Mesh vertices.
    faces : array-like, shape=[n_faces, 3]
        Triangular mesh faces.

    Returns
    -------
    vertex_0 : array-like, shape=[..., n_faces, 3]
        Coordinates of the first vertex of each face.
    vertex_1 : array-like, shape=[..., n_faces, 3]
        Coordinates of the second vertex of each face.
    vertex_2 : array-like, shape=[..., n_faces, 3]
        Coordinates of the third vertex of each face.

    Notes
    -----
    Vertex ordering follows the ordering given by ``faces`` and therefore
    determines the orientation of quantities such as face normals.
    """
    coordinates = compute_face_coordinates(vertices, faces)
    return (
        coordinates[..., 0, :],
        coordinates[..., 1, :],
        coordinates[..., 2, :],
    )


def compute_face_centroids(vertices, faces):
    """Compute face centroids.

    Parameters
    ----------
    vertices : array-like, shape=[..., n_vertices, 3]
        Mesh vertices.
    faces : array-like, shape=[n_faces, 3]
        Triangular mesh faces.

    Returns
    -------
    centroids : array-like, shape=[..., n_faces, 3]
        Face centroids.
    """
    return gs.mean(compute_face_coordinates(vertices, faces), axis=-2)


def compute_face_area_vectors(vertices, faces):
    """Compute oriented face area vectors.

    Parameters
    ----------
    vertices : array-like, shape=[..., n_vertices, 3]
        Mesh vertices..
    faces : array-like, shape=[n_faces, 3]
        Triangular mesh faces.

    Returns
    -------
    area_vectors : array-like, shape=[..., n_faces, 3]
        Oriented area vector of each face. Its direction is determined by
        the vertex ordering in ``faces``, and its norm equals the face area.

    Notes
    -----
    For a face with vertices ``v0``, ``v1``, and ``v2``, the area vector is

    ``0.5 * cross(v1 - v0, v2 - v0)``.

    Normalizing the area vectors gives the unit face normals.
    """
    vertex_0, vertex_1, vertex_2 = compute_face_vertices(vertices, faces)
    return 0.5 * gs.cross(
        vertex_1 - vertex_0,
        vertex_2 - vertex_0,
    )


def compute_face_areas(vertices, faces):
    """Compute face areas.

    Parameters
    ----------
    vertices : array-like, shape=[..., n_vertices, 3]
        Mesh vertices.
    faces : array-like, shape=[n_faces, 3]
        Triangular mesh faces.

    Returns
    -------
    areas : array-like, shape=[..., n_faces]
        Face areas.
    """
    area_vectors = compute_face_area_vectors(vertices, faces)
    return gs.linalg.norm(area_vectors, axis=-1)


def compute_face_normals(vertices, faces):
    """Compute face unit normals.

    Parameters
    ----------
    vertices : array-like, shape=[..., n_vertices, 3]
        Mesh vertices.
    faces : array-like, shape=[n_faces, 3]
        Triangular mesh faces.

    Returns
    -------
    normals : array-like, shape=[..., n_faces, 3]
        Face unit normals. Their orientation is determined by the vertex
        ordering in ``faces``.
    """
    area_vectors = compute_face_area_vectors(vertices, faces)
    areas = gs.linalg.norm(area_vectors, axis=-1)

    return area_vectors / gs.expand_dims(areas, axis=-1)


def compute_face_info(vertices, faces):
    """Compute basic geometric information for triangular faces.

    Parameters
    ----------
    vertices : array-like, shape=[..., n_vertices, 3]
        Mesh vertices.
    faces : array-like, shape=[n_faces, 3]
        Triangular mesh faces.

    Returns
    -------
    centroids : array-like, shape=[..., n_faces, 3]
        Face centroids.
    normals : array-like, shape=[..., n_faces, 3]
        Unit face normals. Their orientation is determined by the vertex
        ordering in ``faces``.
    areas : array-like, shape=[..., n_faces]
        Face areas.
    """
    vertex_0, vertex_1, vertex_2 = compute_face_vertices(vertices, faces)

    centroids = (vertex_0 + vertex_1 + vertex_2) / 3

    area_vectors = 0.5 * gs.cross(
        vertex_1 - vertex_0,
        vertex_2 - vertex_0,
    )
    areas = gs.linalg.norm(area_vectors, axis=-1)
    normals = area_vectors / gs.expand_dims(areas, axis=-1)

    return centroids, normals, areas
