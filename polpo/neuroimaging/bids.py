import re

from polpo.dataset import Dataset

# https://bids.neuroimaging.io/index.html


def select_folders(
    data_dir,
    subject_subset=None,
    session_subset=None,
    subject_regex=r"sub-([A-Za-z0-9]+)",
    session_regex=r"ses-([A-Za-z0-9]+)",
):
    """Select BIDS subject-session folders.

    Parameters
    ----------
    data_dir : pathlib.Path
        Dataset directory.
    subject_subset : array-like
        Subject identifiers to select. If ``None``, all subjects are used.
    session_subset : array-like
        Session identifiers to select. If ``None``, all sessions are used.
    subject_regex : str
        Regular expression used to extract the subject identifier from each
        folder name. The first capture group is used.
    session_regex : str
        Regular expression used to extract the session identifier from each
        folder name. The first capture group is used.

    Returns
    -------
    folders : NestedDataset
        Folder paths indexed by subject and session identifiers.
    """
    folders = {}
    for path in data_dir.glob("sub-*_ses-*"):
        if not path.is_dir() or "sub" not in path.name or "ses" not in path.name:
            continue

        subject_match = re.search(subject_regex, path.name)
        session_match = re.search(session_regex, path.name)
        if subject_match is None or session_match is None:
            continue

        subject_id = subject_match.group(1)
        session_id = session_match.group(1)

        if subject_subset is not None and subject_id not in subject_subset:
            continue

        if session_subset is not None and session_id not in session_subset:
            continue

        folders[(subject_id, session_id)] = path

    return Dataset(folders).nest()


def find_derivative_dir(data_dir, derivative):
    """Find a derivative directory.

    Parameters
    ----------
    data_dir : pathlib.Path
        Dataset directory.
    derivative : str
        Prefix identifying the derivative directory.

    Returns
    -------
    derivative_dir : pathlib.Path
        Matching derivative directory.

    Raises
    ------
    ValueError
        If zero or multiple derivative directories match ``derivative``.
    """
    derivatives_dir = data_dir / "derivatives"

    matches = [
        path
        for path in derivatives_dir.iterdir()
        if path.is_dir() and path.name.startswith(derivative)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one derivative matching {derivative!r}, "
            f"found {[path.name for path in matches]}."
        )

    return matches[0]
