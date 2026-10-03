import numpy as np

from polpo.ext.pymetis import partition_graph

from .ops.topology import compute_vertex_adjacency


def partition_vertices_balanced(faces, n_parts, seed=None):
    """Partition mesh vertices into approximately equal-sized connected patches.

    The mesh is represented by its vertex adjacency graph and partitioned using
    METIS. The partitioning approximately balances the number of vertices
    across patches while minimizing the number of mesh edges crossing between
    patches.

    Parameters
    ----------
    faces : array-like, shape (n_faces, 3)
        Vertex indices of triangular faces.
    n_parts : int
        Number of vertex partitions.
    seed : int or None
        Random seed used for mesh partitioning. If None, a seed is generated at
        runtime.

    Returns
    -------
    labels : ndarray, shape (n_vertices,)
        Partition label assigned to each vertex. Labels are integers in
        ``[0, n_parts)``.
    """
    adj_mat = compute_vertex_adjacency(faces)

    _, labels = partition_graph(adj_mat, n_parts, seed=seed)

    return np.asarray(labels)


def labels_to_vertex_partitions(labels):
    """Convert partition labels to index arrays.

    Parameters
    ----------
    labels : array-like
        Partition label for each item.

    Returns
    -------
    vertex_partitions: list of ndarray
        Indices belonging to each partition.
    """
    return [np.flatnonzero(labels == label) for label in np.unique(labels)]
