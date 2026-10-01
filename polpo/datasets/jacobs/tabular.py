from pathlib import Path

import pandas as pd

from .defaults import DATA_DIR, MATERNAL_PROJECT_FOLDER, PILOT_PROJECT_FOLDER
from .maternal.tabular import load_session_data as load_maternal_session_data
from .pilot.tabular import load_session_data as load_pilot_session_data


def load_session_data(
    data_dir=None,
    subject_subset=None,
    index_by_session=False,
):
    """Load maternal session metadata.

    Parameters
    ----------
    data_dir : path-like
        Data root directory.
    subject_subset : array-like
        Subject identifiers to retain.
    index_by_session : bool
        Whether to index by session when loading a single subject.

    Returns
    -------
    data : pandas.DataFrame
        Maternal session metadata.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    data_dir = Path(data_dir).expanduser()

    include_pilot = subject_subset is None or "01" in subject_subset

    maternal_subset = subject_subset
    if subject_subset is not None:
        maternal_subset = [subject for subject in subject_subset if subject != "01"]

    dfs = []

    if include_pilot:
        dfs.append(
            load_pilot_session_data(
                data_dir=data_dir / PILOT_PROJECT_FOLDER / "rawdata",
                index_by_session=False,
            )
        )

    if maternal_subset is None or maternal_subset:
        dfs.append(
            load_maternal_session_data(
                data_dir=data_dir / MATERNAL_PROJECT_FOLDER / "rawdata",
                subject_subset=maternal_subset,
                index_by_session=False,
            )
        )

    if len(dfs) == 1:
        data = dfs[0]
    else:
        data = pd.concat(dfs, join="inner", ignore_index=True)

    if index_by_session and data["subject"].nunique() == 1:
        data = data.set_index("sessionID")

    return data


def get_session_to_week(
    data_dir=None,
    subject_subset=None,
):
    """Get gestational week indexed by subject and session.

    Parameters
    ----------
    data_dir : path-like
        Directory containing the session data.
    subject_subset : collection of str
        Subject identifiers to include. If ``None``, all subjects are included.

    Returns
    -------
    key_to_week : dict
        Nested dictionary of the form
        ``{subject: {session_id: gest_week}}``.
    """
    data = load_session_data(
        data_dir=data_dir,
        subject_subset=subject_subset,
    )

    return {
        subject: group.set_index("sessionID")["gestWeek"].to_dict()
        for subject, group in data.groupby("subject")
    }
