from pathlib import Path

import polpo.preprocessing.pd as ppd
from polpo.jacobs.defaults import PILOT_DATA_DIR
from polpo.preprocessing import Constant, pipe_to_func


def SessionDataLoader(
    data_dir=None,
    index_by_session=True,
    remove_repeated=True,
):
    if data_dir is None:
        data_dir = PILOT_DATA_DIR / "rawdata"

    filename = "SessionData.csv"
    loader = Constant(Path(data_dir).expanduser() / filename)

    prep_pipe = ppd.UpdateColumnValues(
        column_name="sessionID", func=lambda entry: entry.split("-")[1]
    )
    if remove_repeated:
        prep_pipe += ppd.DfFilter(lambda df: df["sessionID"] == "27", negate=True)

    if index_by_session:
        prep_pipe += ppd.IndexSetter("sessionID", drop=True)

    prep_pipe += ppd.DfInsert(column="subject", value="01")

    return loader + ppd.CsvReader() + prep_pipe


get_session_data = pipe_to_func(SessionDataLoader)
