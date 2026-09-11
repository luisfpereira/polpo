import pymetis

from polpo.seed import resolve_seed


def partition_graph(adj_mat, n_parts, seed=None):
    """Partition a graph into approximately balanced connected parts.

    Parameters
    ----------
    adj_mat : scipy.sparse.csr_matrix, shape (n_vertices, n_vertices)
        Sparse adjacency matrix of the graph.
    n_parts : int
        Number of partitions.
    seed : int or None
        Random seed used for mesh partitioning. If None, a seed is generated at
        runtime.

    Returns
    -------
    n_edge_cuts : int
        Number of graph edges crossing between different partitions.
    labels : array-like, shape (n_vertices,)
        Partition label assigned to each vertex.
    """
    seed = resolve_seed(seed)

    adjacency = pymetis.CSRAdjacency(
        adj_mat.indptr,
        adj_mat.indices,
    )

    n_edge_cuts, labels = pymetis.part_graph(
        n_parts,
        adjacency=adjacency,
        options=pymetis.Options(contig=True, seed=seed),
    )
    return n_edge_cuts, labels
