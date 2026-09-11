import polpo.preprocessing.dict as ppdict
from polpo.jacobs.path import FoldersSelector
from polpo.preprocessing import Constant
from polpo.preprocessing.mri import MriImageLoader
from polpo.preprocessing.path import FileFinder, IsFileType
from polpo.preprocessing.str import StartsWith

from .defaults import DATA_DIR


def AshsSegmentationFinder(left=True):
    prefix = "left_" if left else "right_"
    return FileFinder(
        rules=[
            IsFileType("nii.gz"),
            StartsWith(prefix),
        ]
    )


def AshsSegmentationsLoader(
    data_dir=None,
    session_subset=None,
    left=True,
    as_image=False,
):
    """Create pipeline to load segmented mri filenames.

    Parameters
    ----------
    data_dir : str
        Directory where to store data.
    subset : array-like
        Subset of sessions to load. If `None`, loads all.
    subject_id : str
        Identification of the subject. If None, assumes pilot.
    as_image : bool
        Whether to load file as image.

    Returns
    -------
    pipe : Pipeline
        Pipeline to load segmented mri filenames.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    folders_selector = (
        Constant(data_dir)
        + FoldersSelector(
            subject_subset=("01",),
            session_subset=session_subset,
            derivative="ashs",
        )
        + ppdict.ExtractUniqueKey()
    )

    if left and (session_subset is None or "26" in session_subset):
        # issue with "26" file
        folders_selector += ppdict.RemoveKeys(["26"])

    image_selector = AshsSegmentationFinder(left=left)

    if as_image:
        image_selector += MriImageLoader()

    return folders_selector + ppdict.DictMap(image_selector)
