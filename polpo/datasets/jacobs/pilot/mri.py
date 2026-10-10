from pathlib import Path

from polpo.neuroimaging.mri import load_image

from ..defaults import PILOT_DATA_DIR
from .path import select_folders


def select_ashs_segmentation(path, left=True):
    """Select an ASHS segmentation file.

    Parameters
    ----------
    path : pathlib.Path
        Directory containing ASHS segmentation files.
    left : bool
        Whether to select the left or right hemisphere.

    Returns
    -------
    path : pathlib.Path
        Selected segmentation file.

    Raises
    ------
    ValueError
        If exactly one matching file is not found.
    """
    prefix = "left_" if left else "right_"
    paths = sorted(path.glob(f"{prefix}*.nii.gz"))

    if len(paths) != 1:
        raise ValueError(
            f"Expected one ASHS segmentation in {path}, found {len(paths)}."
        )

    return paths[0]


def load_ashs_segmentations(
    data_dir=None,
    session_subset=None,
    left=True,
    as_image=False,
):
    """Load pilot ASHS segmentations indexed by subject and session.

    Parameters
    ----------
    data_dir : path-like
        Pilot-project directory.
    session_subset : iterable of str or None
        Session identifiers to select.
    left : bool
        Whether to select left or right hemisphere segmentations.
    as_image : bool
        Whether to load segmentation images instead of returning paths.

    Returns
    -------
    segmentations : NestedDataset
        Segmentations indexed by subject and session.
    """
    if data_dir is None:
        data_dir = PILOT_DATA_DIR

    folders = select_folders(
        Path(data_dir),
        derivative="ashs",
        session_subset=session_subset,
    )

    if left:
        # missing or problematic left segmentation for session 26
        folders = folders.drop_inner({"01": ["26"]})

    def _load(path):
        path = select_ashs_segmentation(path, left=left)
        return load_image(path) if as_image else path

    return folders.map_values(_load)
