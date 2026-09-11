from pathlib import Path

import polpo.preprocessing.dict as ppdict
from polpo.preprocessing import (
    ContainsAll,
    FilteredGroupBy,
    Map,
)
from polpo.preprocessing.path import (
    ExpandUser,
    FileFinder,
    PathShortener,
)
from polpo.preprocessing.str import RegexGroupFinder, StartsWith

# https://bids.neuroimaging.io/index.html


def FoldersSelector(
    subject_subset=None,
    session_subset=None,
    session_sorter=True,
    subject_regex=r"sub-([A-Za-z0-9]+)",
    session_regex=r"ses-([A-Za-z0-9]+)",
):
    """Create pipeline to select BIDS subject-session folders.

    The pipeline takes a directory as input, finds folders whose paths
    contain both ``"sub"`` and ``"ses"``, and returns a nested dictionary:

    ``output[subject_id][session_id] -> folder_path``

    Parameters
    ----------
    subject_subset : array-like, optional
        Subject identifiers to select. If ``None``, all subjects are used.
    session_subset : array-like, optional
        Session identifiers to select. If ``None``, all sessions are used.
    session_sorter : callable or bool or None, optional
        Function used to sort session identifiers. If ``True``, sessions are
        sorted by their identifier. If ``None``, sorting is skipped.
    subject_regex : str, optional
        Regular expression used to extract the subject identifier from each
        folder path. The first capture group is used.
    session_regex : str, optional
        Regular expression used to extract the session identifier from each
        folder path. The first capture group is used.

    Returns
    -------
    pipe : Pipeline
        Pipeline mapping an input directory to a nested dictionary of folder
        paths indexed by subject and session identifiers.
    """
    if session_sorter is True:
        session_sorter = lambda x: x

    folders_selector = ExpandUser() + FileFinder(
        rules=ContainsAll(["sub", "ses"]), as_list=True
    )

    path_to_subject_id = PathShortener() + RegexGroupFinder(subject_regex)
    path_to_session_id = PathShortener() + RegexGroupFinder(session_regex)

    folders_selector += FilteredGroupBy(path_to_subject_id, subset=subject_subset)

    folders_selector += ppdict.DictMap(
        ppdict.HashWithIncoming(
            key_step=Map(path_to_session_id),
            key_subset=session_subset,
            key_sorter=session_sorter,
        )
    )

    return folders_selector


def DerFolderSelector(derivative):
    return (
        (lambda path: Path(path) / "derivatives")
        + ExpandUser()
        + FileFinder(rules=StartsWith(derivative))
    )


def DerSessionFolderSelector(
    derivative,
    subject_subset=None,
    session_subset=None,
    session_sorter=True,
    subject_regex=r"sub-([A-Za-z0-9]+)",
    session_regex=r"ses-([A-Za-z0-9]+)",
):
    """Create pipeline to select derivative session folders.

    The pipeline takes a dataset root directory as input and returns
    a nested dictionary indexed by subject and session identifiers:

    ``output[subject_id][session_id] -> folder_path``

    Parameters
    ----------
    derivative : str
        Name of the derivative folder (e.g. ``"fsl_first"``,
        ``"fastsurfer-long"``).
    subject_subset : array-like
        Subject identifiers to select. If ``None``, all subjects are used.
    session_subset : array-like
        Session identifiers to select. If ``None``, all sessions are used.


    Returns
    -------
    pipe : Pipeline
        Pipeline mapping a dataset root directory to a nested dictionary
        of derivative session folder paths indexed by subject and session.
    """
    return DerFolderSelector(derivative) + FoldersSelector(
        subject_subset,
        session_subset=session_subset,
        session_sorter=session_sorter,
        subject_regex=subject_regex,
        session_regex=session_regex,
    )
