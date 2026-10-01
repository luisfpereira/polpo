import pandas as pd

from polpo.datasets.jacobs.defaults import PILOT_DATA_DIR


def load_session_data(
    data_dir=None,
    index_by_session=True,
    remove_repeated=True,
):
    """Load pilot session metadata.

    Parameters
    ----------
    data_dir : path-like
        Directory containing ``SessionData.csv``.
    index_by_session : bool
        Whether to index rows by session identifier.
    remove_repeated : bool
        Whether to remove the repeated session ``27``.

    Returns
    -------
    data : pandas.DataFrame
        Session metadata.
    """
    if data_dir is None:
        data_dir = PILOT_DATA_DIR / "rawdata"

    path = data_dir / "SessionData.csv"
    data = pd.read_csv(path)
    data.drop(columns="trimester", inplace=True)

    data["sessionID"] = data["sessionID"].str.removeprefix("ses-")

    if remove_repeated:
        data = data[data["sessionID"] != "27"]

    data.insert(0, "subject", "01")

    if index_by_session:
        data = data.set_index("sessionID")

    return data
