import numpy as np


def select_labels(array, labels, binary=True, encoding=None):
    """Select labels from an array.

    Parameters
    ----------
    array : array-like
        Array containing labeled values.
    labels : iterable
        Values to select.
    binary : bool
        Whether to return a boolean mask or preserve selected values,
        setting all others to zero.
    encoding : callable or None
        Function mapping labels to array values. If None, labels are used directly.

    Returns
    -------
    array : ndarray
        Array with the same shape as the input.
    """
    if encoding is not None:
        labels = [encoding(label) for label in labels]

    mask = np.isin(array, labels)
    return mask if binary else np.where(mask, array, 0)


def split_labels(array, labels, binary=True, encoding=None):
    """Split an array into label-specific selections.

    Parameters
    ----------
    array : array-like
        Array containing labeled values.
    labels : iterable
        Values to select individually.
    binary : bool
        Whether to return boolean masks or preserve selected values,
        setting all others to zero.
    encoding : callable or None
        Function mapping labels to array values. If None, labels are used directly.

    Returns
    -------
    arrays: dict
        Mapping from each label to its corresponding array.
    """
    return {
        label: select_labels(array, [label], binary=binary, encoding=encoding)
        for label in labels
    }
