from polpo.dataset import NestedDataset

from .defaults import MATERNAL_PROJECT_FOLDER, PILOT_PROJECT_FOLDER
from .maternal.path import select_folders as select_maternal_folders
from .pilot.path import select_folders as select_pilot_folders
from .utils import get_subject_ids, validate_subject_subset


def _split_subject_subset(subject_subset=None):
    """Split subject identifiers into pilot and main-project subsets.

    Parameters
    ----------
    subject_subset : array-like
        Subject identifiers to split. If ``None``, all available subjects
        are used.

    Returns
    -------
    pilot_subset : list
        Pilot subject identifiers.
    subject_subset : list
        Main-project subject identifiers.

    Raises
    ------
    ValueError
        If a subject identifier is not available.
    """
    if subject_subset is None:
        subject_subset = get_subject_ids()

    validate_subject_subset(subject_subset)

    pilot_subset = ["01"] if "01" in subject_subset else []
    subject_subset = [subject_id for subject_id in subject_subset if subject_id != "01"]

    return pilot_subset, subject_subset


def select_folders(
    data_dir,
    derivative,
    subject_subset=None,
    session_subset=None,
):
    """Select derivative folders for maternal-study subjects.

    Data are collected from the pilot and main projects as needed and
    returned in a single dataset.

    Parameters
    ----------
    data_dir : pathlib.Path
        Maternal-study data directory.
    derivative : str
        Prefix identifying the derivative directory.
    subject_subset : array-like
        Subject identifiers to select. If ``None``, all available subjects
        are used.
    session_subset : array-like
        Session identifiers to select. If ``None``, all sessions are used.

    Returns
    -------
    folders : NestedDataset
        Folder paths indexed by subject and session identifiers.

    Raises
    ------
    ValueError
        If sessions are filtered while selecting subjects from both the
        pilot and main projects.
    """
    pilot_subset, subject_subset = _split_subject_subset(subject_subset)

    if len(pilot_subset) and len(subject_subset) and session_subset is not None:
        raise ValueError("Can't filter sessions if pilot included")

    folders = []
    if len(pilot_subset):
        folders.append(
            select_pilot_folders(
                data_dir / PILOT_PROJECT_FOLDER,
                derivative,
                session_subset=session_subset,
                remove_repeated=True,
            )
        )

    if len(subject_subset):
        folders.append(
            select_maternal_folders(
                data_dir / MATERNAL_PROJECT_FOLDER,
                derivative,
                subject_subset=subject_subset,
                session_subset=session_subset,
            )
        )

    return NestedDataset.merge_many(folders).sort_keys()
