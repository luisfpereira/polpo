import pandas as pd

from polpo.datasets.jacobs.defaults import MATERNAL_DATA_DIR


def load_session_data(
    data_dir=None,
    subject_subset=None,
    index_by_session=False,
):
    """Load maternal session metadata.

    Parameters
    ----------
    data_dir : path-like
        Directory containing ``SessionData.csv``.
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
        data_dir = MATERNAL_DATA_DIR / "rawdata"

    path = data_dir / "SessionData.csv"
    data = pd.read_csv(path)
    data.drop(columns="trimester", inplace=True)

    data["sessionID"] = data["sessionID"].str.removeprefix("ses-")
    data["subject"] = data["subject"].str.removeprefix("sub-")

    if subject_subset is not None:
        data = data[data["subject"].isin(subject_subset)]

    mask = (data["sessionID"] == "post6") & (data["subject"] == "1009B")
    data.loc[mask, "postDays"] = 128.0

    if index_by_session and subject_subset is not None and len(subject_subset) == 1:
        data = data.set_index("sessionID")

    return data
