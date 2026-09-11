import polpo.preprocessing.dict as ppdict
from polpo.neuroi.mri import SubcorticalSegmentationFinder
from polpo.preprocessing import Constant, pipe_to_func
from polpo.preprocessing.mri import MriImageLoader

from .defaults import DATA_DIR
from .path import FoldersSelector


def SubcorticalSegmentationsLoader(
    derivative,
    data_dir=None,
    subject_subset=None,
    session_subset=None,
    as_image=False,
):
    """Create pipeline to load segmented mri filenames.

    Parameters
    ----------
    derivative : str
        Tool used to generate derivatives.
        One of the following: "fsl*", "fast*".
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

    folders_selector = Constant(data_dir) + FoldersSelector(
        subject_subset=subject_subset,
        session_subset=session_subset,
        derivative=derivative,
    )

    image_selector = SubcorticalSegmentationFinder(derivative)

    if as_image:
        image_selector += MriImageLoader()

    return folders_selector + ppdict.NestedDictMap(image_selector)


load_subcortical_segmentations = pipe_to_func(SubcorticalSegmentationsLoader)
