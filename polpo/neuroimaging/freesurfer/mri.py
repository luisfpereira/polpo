from polpo.ext.nibabel import compute_label_volumes
from polpo.ext.numpy.labels import select_labels, split_labels
from polpo.neuroimaging._dispatch import (
    compute_segmentation_volumes as _segmentation_volume_dispatcher,
)
from polpo.neuroimaging._dispatch import (
    select_segmentation_path as _segmentation_path_dispatcher,
)

from .naming import name_to_aseg_id


@_segmentation_path_dispatcher.register("fast")
@_segmentation_path_dispatcher.register("free")
def select_segmentation_path(path, auto=False):
    """Select a FreeSurfer subcortical segmentation file.

    Parameters
    ----------
    path : path-like
        Path to the FreeSurfer subject directory.
    auto : bool
        Whether to select the automatically generated segmentation
        (`aseg.auto.mgz`) instead of the final segmentation (`aseg.mgz`).

    Returns
    -------
    pathlib.Path
        Path to the selected segmentation file.

    Raises
    ------
    FileNotFoundError
        If the segmentation file does not exist.
    """
    filename = "aseg.auto.mgz" if auto else "aseg.mgz"
    segmentation_path = path / "mri" / filename

    if not segmentation_path.is_file():
        raise FileNotFoundError(segmentation_path)

    return segmentation_path


def select_segmentation_labels(array, labels, binary=True):
    """Select FreeSurfer aseg labels by anatomical name."""
    return select_labels(array, labels, binary=binary, encoding=name_to_aseg_id)


def split_segmentation_labels(array, labels, binary=True):
    """Split FreeSurfer aseg labels by anatomical name."""
    return split_labels(array, labels, binary=binary, encoding=name_to_aseg_id)


@_segmentation_volume_dispatcher.register("fast")
@_segmentation_volume_dispatcher.register("free")
def compute_segmentation_volumes(path, labels):
    return compute_label_volumes(path, labels, encoding=name_to_aseg_id)
