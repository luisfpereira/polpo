import numpy as np


def normalize_edges(edges, return_permutation=False):
    edges = np.sort(edges, axis=1)
    permutation = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[permutation]

    if return_permutation:
        return edges, permutation

    return edges
