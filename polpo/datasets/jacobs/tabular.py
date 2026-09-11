from pathlib import Path

import numpy as np
import pandas as pd

import polpo.preprocessing.pd as ppd
import polpo.utils as putils
from polpo.preprocessing import BranchingPipeline, Constant, pipe_to_func

from .defaults import DATA_DIR, PILOT_PROJECT_FOLDER, PROJECT_FOLDER
from .pilot.tabular import SessionDataLoader as PilotSessionDataLoader


def _remove_prefix(entry, sep="-"):
    return entry.split(sep)[1]


def _SessionDataLoader(
    data_dir=None,
    subject_subset=None,
    index_by_session=False,
):
    """Create pipeline to load maternal csv data.

    Parameters
    ----------
    data_dir : str
        Data root dir.
    subject_subset : array-like
        Id of the subjects. If None, assumes all.
    index_by_session : bool
        Whether to index the dataframe by session.
        Only applies if one subject.

    Returns
    -------
    pipe : Pipeline
        Pipeline to load maternal csv data without the pilot.
    """
    if data_dir is None:
        data_dir = DATA_DIR / PROJECT_FOLDER / "rawdata"

    filename = "SessionData.csv"
    loader = Constant(Path(data_dir).expanduser() / filename)

    prep_pipe = ppd.UpdateColumnValues(
        column_name="sessionID",
        func=_remove_prefix,
    )
    prep_pipe += ppd.UpdateColumnValues(
        column_name="subject",
        func=_remove_prefix,
    )

    if subject_subset is not None:
        prep_pipe += ppd.DfIsInFilter("subject", subject_subset)

    if index_by_session and (subject_subset is not None and len(subject_subset) == 1):
        prep_pipe += ppd.IndexSetter("sessionID", drop=True)

    prep_pipe += ppd.SetColumnValueWhere(
        lambda df: (df["sessionID"] == "post6") & (df["subject"] == "1009B"),
        "postDays",
        128.0,
    )

    return loader + ppd.CsvReader() + prep_pipe


def SessionDataLoader(
    data_dir=None,
    subject_subset=None,
    index_by_session=False,
):
    """Create pipeline to load maternal csv data.

    Parameters
    ----------
    data_dir : str
        Data root dir.
    subject_subset : array-like
        Id of the subjects. If None, assumes all.
        One of the following: "01", "1001B", "1004B".
        If pilot and other, loads only common columns.
    index_by_session : bool
        Whether to index the dataframe by session.
        Only applies if one subject.

    Returns
    -------
    pipe : Pipeline
        Pipeline to load maternal csv data.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    data_dir = Path(data_dir).expanduser()

    pilot_pipe = None
    if subject_subset is None or "01" in subject_subset:
        pilot_pipe = PilotSessionDataLoader(
            data_dir=data_dir / PILOT_PROJECT_FOLDER / "rawdata",
            index_by_session=index_by_session and len(subject_subset) == 1,
        )

    if pilot_pipe and (subject_subset is not None and len(subject_subset) == 1):
        return pilot_pipe

    if subject_subset is not None and "01" in subject_subset:
        subject_subset = subject_subset.copy()
        subject_subset.remove("01")

    pipe = _SessionDataLoader(
        data_dir=data_dir / PROJECT_FOLDER / "rawdata",
        subject_subset=subject_subset,
        index_by_session=index_by_session and len(subject_subset) == 1,
    )

    if pilot_pipe is None:
        return pipe

    return BranchingPipeline(
        branches=[pilot_pipe, pipe],
        merger=lambda dfs: pd.concat(dfs, join="inner", ignore_index=True),
    )


def get_key_to_week(
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
    df = get_session_data(data_dir=data_dir, subject_subset=subject_subset)

    return putils.df_to_nested_dict(
        df, outer_key="subject", inner_key="sessionID", value_col="gestWeek"
    )


def get_birth_week(
    data_dir=None,
    subject_subset=None,
    na_value=42.0,
):
    """Get birth gestational week for each subject.

    Birth week is inferred from the first postpartum session, ``post1``, as

    ``floor(gestWeek - postDays // 7)``.

    Subject ``"01"`` is handled manually with birth week ``40.0`` [PTC2024]_.

    Parameters
    ----------
    data_dir : path-like
        Directory containing the session data.
    subject_subset : collection of str
        Subject identifiers to include. If ``None``, all subjects are included.
    na_value : float or None
        Value used to fill missing ``gestWeek`` or ``postDays`` values.
        If ``None``, rows with missing values in these columns are dropped.

    Returns
    -------
    birth_week : dict
        Dictionary of the form ``{subject: birth_week}``.

    References
    ----------
    .. [PTC2024] Pritschet, L., Taylor, C.M., Cossio, D., Faskowitz, J.,
        Santander, T., Handwerker, D.A., Grotzinger, H., Layher, E., Chrastil,
        E.R., Jacobs, E.G., 2024. Neuroanatomical changes observed over the
        course of a human pregnancy. Nat Neurosci 27, 2253–2260.
        https://doi.org/10.1038/s41593-024-01741-0
    """
    birth_week = {}
    if subject_subset is None or "01" in subject_subset:
        birth_week["01"] = 40.0

    df = _SessionDataLoader(
        data_dir=data_dir / PROJECT_FOLDER / "rawdata", subject_subset=subject_subset
    )()

    df_ = df.loc[df["sessionID"] == "post1"].copy()
    if na_value is None:
        df_.dropna(inplace=True)
    else:
        df_.fillna(na_value, inplace=True)

    df_["partumGestWeek"] = np.floor(df_["gestWeek"] - df_["postDays"] // 7)

    birth_week_ = ppd.ColumnPairToDict("subject", "partumGestWeek")(df_)

    birth_week.update(birth_week_)

    return birth_week


def get_key_to_birth_week(
    data_dir=None,
    subject_subset=None,
    na_value=42.0,
):
    """Get session timing relative to birth week.

    Parameters
    ----------
    data_dir : path-like
        Directory containing the session data.
    subject_subset : collection of str
        Subject identifiers to include. If ``None``, all subjects are included.
    na_value : float or None
        Value used when estimating birth week from missing postpartum data.
        See :func:`get_birth_week`.

    Returns
    -------
    key_to_birth : dict
        Nested dictionary of the form
        ``{subject: {session_id: weeks_from_birth}}``.
    """
    key_to_birth = {}
    birth_week = get_birth_week(data_dir, subject_subset, na_value)

    for subj_id, session_dict in get_key_to_week(data_dir, subject_subset).items():
        birth = birth_week.get(subj_id)
        key_to_birth[subj_id] = {
            key: value - birth for key, value in session_dict.items()
        }

    return key_to_birth


get_session_data = pipe_to_func(SessionDataLoader)
