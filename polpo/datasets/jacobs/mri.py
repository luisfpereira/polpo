from polpo.neuroimaging.mri import load_image, select_segmentation_path

from .defaults import DATA_DIR
from .path import select_folders


def load_subcortical_segmentations(
    derivative,
    data_dir=None,
    subject_subset=None,
    session_subset=None,
    as_image=False,
):
    """Load subcortical segmentations grouped by subject and session.

    Parameters
    ----------
    derivative : str
        Prefix identifying the derivative directory.
    data_dir : str or pathlib.Path
        Dataset root directory.
    subject_subset : array-like
        Subject identifiers to select. If None, all subjects are used.
    session_subset : array-like
        Session identifiers to select. If None, all sessions are used.
    as_image : bool
        Whether to load segmentation paths as image arrays.

    Returns
    -------
    dataset : NestedDataset
        Dataset indexed by subject and session, containing segmentation
        paths or image arrays.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    folders = select_folders(
        data_dir,
        derivative,
        subject_subset=subject_subset,
        session_subset=session_subset,
    )

    def _load_segmentation(path):
        path = select_segmentation_path(path, derivative=derivative)
        return load_image(path) if as_image else path

    return folders.map_values(_load_segmentation)
