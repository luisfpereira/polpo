import re

from polpo.neuroimaging.bids import find_derivative_dir
from polpo.neuroimaging.bids import select_folders as select_bids_folders


def _session_sort_key(session_id):
    """Return a sorting key for session identifiers.

    Parameters
    ----------
    session_id : str
        Session identifier ending in an integer.

    Returns
    -------
    sort_key : tuple
        Session prefix and trailing integer.
    """
    prefix, index = re.match(r"(.*?)(\d+)$", session_id).groups()
    return prefix, int(index)


def select_folders(
    data_dir,
    derivative,
    subject_subset,
    session_subset=None,
):
    """Select derivative folders for main-project subjects.

    Parameters
    ----------
    data_dir : pathlib.Path
        Main-project data directory.
    derivative : str
        Prefix identifying the derivative directory.
    subject_subset : array-like
        Subject identifiers to select.
    session_subset : array-like
        Session identifiers to select. If ``None``, all sessions are used.

    Returns
    -------
    folders : NestedDataset
        Folder paths indexed by subject and session identifiers.
    """
    path = find_derivative_dir(data_dir, derivative)

    folders = select_bids_folders(
        path,
        subject_subset,
        session_subset,
    ).sort_inner_keys(_session_sort_key)

    excluded_sessions = {
        "1011B": {"pre1", "pre2"},
    }

    return folders.drop_inner(excluded_sessions)
