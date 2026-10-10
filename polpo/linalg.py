from math import isqrt

import numpy as np


def sym_to_triu_vec(mat, k=1):
    """Extract the upper triangle of a symmetric matrix.

    Parameters
    ----------
    mat : array-like, shape (n, n)
        Symmetric matrix.
    k : int
        Diagonal offset. Use 0 to include the main diagonal and 1
        to exclude it.

    Returns
    -------
    vec : ndarray, shape (m,)
        Flattened upper triangle, ordered row-wise.
        For k=0, m=n(n+1)/2; for k=1, m=n(n-1)/2.
    """
    return mat[np.triu_indices(len(mat), k=k)]


def triu_vec_to_sym(vec, includes_diag=False):
    """Reconstruct a symmetric matrix from its upper triangle.

    Parameters
    ----------
    vec : array-like, shape (m,)
        Flattened upper triangle, ordered row-wise.
        Its length must correspond to a triangular number.
    includes_diag : bool
        Whether the vector includes the main diagonal.
        If False, the reconstructed diagonal is zero.

    Returns
    -------
    mat : ndarray, shape (n, n)
        Reconstructed symmetric matrix, where m=n(n+1)/2
        if the diagonal is included, and m=n(n-1)/2 otherwise.
    """
    k = 0 if includes_diag else 1
    n = (isqrt(8 * vec.size + 1) + 1 - 2 * includes_diag) // 2

    mat = np.zeros((n, n), dtype=vec.dtype)

    i, j = np.triu_indices(n, k=k)
    mat[i, j] = vec
    mat[j, i] = vec

    return mat


def get_diag_blocks_by_size(mat, sizes):
    """Extract diagonal blocks of specified sizes from a matrix.

    Parameters
    ----------
    mat : array-like, shape (n, n)
        Matrix to extract blocks from.
    sizes : sequence of int
        Sizes of consecutive diagonal blocks, expected to sum to n.

    Returns
    -------
    blocks : list of ndarray
        Diagonal blocks, where block i has shape (sizes[i], sizes[i]).
        Blocks are views into the original matrix.
    """
    indices = np.r_[0, np.cumsum(sizes)]

    return [mat[start:end, start:end] for start, end in zip(indices[:-1], indices[1:])]


def permute_by_row_norm(mat, descending=True):
    """Permute rows and columns of a square matrix by row norm.

    Parameters
    ----------
    mat : ndarray, shape (n, n)
        Matrix to permute.
    descending : bool
        Whether to sort by decreasing row norm.

    Returns
    -------
    permuted_mat : ndarray, shape (n, n)
        Matrix with both axes permuted.
    indices : ndarray, shape (n,)
        Indices defining the permutation.
    """
    row_norms = np.linalg.norm(mat, axis=-1)

    signal = -1.0 if descending else 1.0
    sorted_idx = np.argsort(signal * row_norms, axis=-1)

    perm_mat = mat[np.ix_(sorted_idx, sorted_idx)]

    return perm_mat, sorted_idx
