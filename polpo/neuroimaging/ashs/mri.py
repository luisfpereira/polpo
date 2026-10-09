from polpo.ext.numpy.labels import select_labels, split_labels

from .naming import NAME_TO_ASHS_ID


def select_segmentation_labels(array, labels, binary=True):
    """Select ASHS labels by anatomical name."""
    return select_labels(
        array, labels, binary=binary, encoding=NAME_TO_ASHS_ID.__getitem__
    )


def split_segmentation_labels(array, labels, binary=True):
    """Split ASHS labels by anatomical name."""
    return split_labels(
        array, labels, binary=binary, encoding=NAME_TO_ASHS_ID.__getitem__
    )
