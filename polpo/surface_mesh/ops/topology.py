import geomstats.backend as gs
import numpy as np
from scipy import sparse


def normalize_edges(edges, return_permutation=False):
    """Normalize the ordering of undirected mesh edges.

    Each edge is first ordered by vertex index, then all edges are sorted
    lexicographically.

    Parameters
    ----------
    edges : array-like, shape=(n_edges, 2)
        Vertex indices defining the mesh edges.
    return_permutation : bool
        Whether to also return the permutation used to sort the edges.

    Returns
    -------
    edges : array-like, shape=(n_edges, 2)
        Normalized and lexicographically sorted edges.
    permutation : array-like, shape=(n_edges,)
        Permutation used to sort the edges. Returned only when
        ``return_permutation`` is ``True``.
    """
    edges = np.sort(edges, axis=1)
    permutation = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[permutation]

    if return_permutation:
        return edges, permutation

    return edges


def compute_edges(faces, *, unique=False, oriented=False):
    """Compute edges from triangular mesh faces.

    Parameters
    ----------
    faces : array-like, shape (n_faces, 3)
        Vertex indices of triangular faces.
    oriented : bool, default=False
        If True, preserve the orientation induced by the ordering of
        vertices in each face. If False, treat edges as undirected.
    unique : bool, default=True
        If True, remove duplicate edges.

    Returns
    -------
    edges : ndarray, shape (n_edges, 2)
        Vertex indices defining the mesh edges. If ``unique=False``,
        ``n_edges = 3 * n_faces``.
    """
    edges = gs.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )

    if not oriented:
        edges.sort(axis=1)

    if unique:
        edges = gs.unique(edges, axis=0)

    return edges


def compute_vertex_adjacency(faces):
    """Compute the sparse vertex adjacency matrix of a triangular mesh.

    Parameters
    ----------
    faces : array-like, shape (n_faces, 3)
        Vertex indices of triangular faces.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix, shape (n_vertices, n_vertices)
        Symmetric boolean adjacency matrix, where ``adjacency[i, j]`` is
        True when vertices ``i`` and ``j`` share a mesh edge.
    """
    edges = compute_edges(faces, unique=False)

    n_vertices = faces.max() + 1
    data = np.ones(edges.shape[0], dtype=bool)
    adj_mat = sparse.coo_matrix(
        (
            data,
            (edges[:, 0], edges[:, 1]),
        ),
        shape=(n_vertices, n_vertices),
    ).tocsr()

    return adj_mat + adj_mat.T


def compute_one_ring_neighbors(faces):
    """Convert triangular faces to vertex adjacency lists.

    Parameters
    ----------
    faces : array-like, shape (n_faces, 3)
        Triangle vertex indices.

    Returns
    -------
    adjacency : list of list of int
        Vertex adjacency lists. ``adjacency[i]`` contains the indices
        of vertices connected to vertex ``i`` by an edge.
    """
    adj_mat = compute_vertex_adjacency(faces)

    return [
        adj_mat.indices[start:end].tolist()
        for start, end in zip(adj_mat.indptr[:-1], adj_mat.indptr[1:])
    ]
