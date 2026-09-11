def batch_slices(array, n_nonbatch_dims):
    """Return full slices over the batch dimensions of an array.

    Parameters
    ----------
    array : array-like
        Input array.
    n_nonbatch_dims : int
        Number of trailing dimensions that do not belong to the batch shape.

    Returns
    -------
    slices : tuple[slice]
        One full slice for each batch dimension.

    Examples
    --------
    An array of shape ``(4, 5, 100, 3)`` with two non-batch dimensions
    has batch shape ``(4, 5)``.

    >>> batch_slices(array, n_nonbatch_dims=2)
    (slice(None), slice(None))
    """
    n_batch_dims = array.ndim - n_nonbatch_dims
    return (slice(None),) * n_batch_dims
