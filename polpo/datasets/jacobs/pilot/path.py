from polpo.neuroimaging.bids import find_derivative_dir, select_folders
from polpo.neuroimaging.bids import select_folders as select_bids_folders


def select_folders(
    data_dir,
    derivative,
    session_subset=None,
    remove_repeated=True,
):
    """Select derivative folders for the pilot subject.

    Parameters
    ----------
    data_dir : pathlib.Path
        Pilot-project data directory.
    derivative : str
        Prefix identifying the derivative directory.
    session_subset : array-like
        Session identifiers to select. If ``None``, all sessions are used.
    remove_repeated : bool
        Whether to remove repeated sessions.

    Returns
    -------
    folders : NestedDataset
        Folder paths indexed by subject and session identifiers.
    """
    path = find_derivative_dir(data_dir, derivative)
    subject_subset = "01"

    folders = select_bids_folders(
        path,
        subject_subset,
        session_subset,
    ).sort_inner_keys(key=int)

    if remove_repeated:
        # same session metadata as 2
        folders = folders.filter_keys(
            lambda _, session_id: session_id != "27",
        )

    return folders
