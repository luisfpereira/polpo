import numpy as np
import polars as pl

from polpo.utils import unnest_list

from .naming import get_all_subcortical_structs, name_to_aseg_id


def _index2cols(header, index, output="LogJacs"):
    """Find columns associated with an ENIGMA structure index.

    Parameters
    ----------
    header : polars.DataFrame
        DataFrame containing ENIGMA output columns.
    index : int
        FreeSurfer ASEG structure identifier.
    output : str
        Output type.

    Returns
    -------
    columns : list of str
        Matching column names.
    """
    return [c for c in header.columns if c.startswith(f"{output}_{index}_")]


def _struct_subset2cols(header, struct_subset, output="LogJacs"):
    """Map structure names to their ENIGMA output columns.

    Parameters
    ----------
    header : polars.DataFrame
        DataFrame containing ENIGMA output columns.
    struct_subset : iterable of str or None
        Structures to select. If None, all subcortical structures are used.
    output : str
        Output type.

    Returns
    -------
    name2cols : dict
        Mapping from structure names to lists of column names.
    """
    if struct_subset is None:
        struct_subset = get_all_subcortical_structs()

    enigma_indices = [name_to_aseg_id(struct) for struct in struct_subset]

    name2cols = {}
    for name, index in zip(struct_subset, enigma_indices):
        name2cols[name] = _index2cols(header, index, output=output)

    return name2cols


def load_session_output(filename, struct_subset=None, output="LogJacs"):
    """Load structure-wise outputs from an ENIGMA session CSV.

    Parameters
    ----------
    filename : path-like
        Path to the ENIGMA output CSV file.
    struct_subset : iterable of str or None
        Structures to select. If None, all subcortical structures are used.
    output : str
        Output type.

    Returns
    -------
    data : dict
        Mapping from structure names to NumPy arrays of output values.
        Singleton dimensions are removed.
    """
    df = pl.read_csv(filename)
    name2cols = _struct_subset2cols(df, struct_subset, output=output)

    data = {}
    for name, cols in name2cols.items():
        data[name] = df[cols].to_numpy().squeeze()

    return data


def load_session_outputs(filenames, struct_subset=None, output="LogJacs"):
    """Load structure-wise outputs from multiple ENIGMA session CSV files.

    Parameters
    ----------
    filenames : iterable of path-like
        Paths to ENIGMA output CSV files.
    struct_subset : iterable of str or None
        Structures to select. If None, all subcortical structures are used.
    output : str
        Output type.

    Returns
    -------
    data : list of dict
        Structure-wise output arrays for each file, preserving input order.
    """
    name2cols = None
    all_cols = None

    data = []
    for filename in filenames:
        df = pl.read_csv(filename, columns=all_cols)
        if name2cols is None:
            name2cols = _struct_subset2cols(df, struct_subset, output=output)
            all_cols = unnest_list(name2cols.values())

        data_ = {}
        for struct_name, cols in name2cols.items():
            data_[struct_name] = df[cols].to_numpy().squeeze()

        data.append(data_)

    return data


def load_output(
    filename,
    subject_subset=None,
    session_subset=None,
    struct_subset=None,
    output="LogJacs",
):
    """Load ENIGMA outputs organized by subject, session, and structure.

    Parameters
    ----------
    filename : path-like
        Path to an ENIGMA ``subjects_file`` CSV.
    subject_subset : iterable of str or None
        Subject identifiers to retain.
    session_subset : iterable of str or None
        Session identifiers to retain.
    struct_subset : iterable of str or None
        Structure names to retain. If None, all subcortical structures
        are used.
    output : {"LogJacs", "thick"}
        ENIGMA output type.

    Returns
    -------
    data : dict
        Nested mapping from subject identifiers to session identifiers
        to structure names. Each leaf contains a NumPy array of
        vertex-wise output values.

    Raises
    ------
    ValueError
        If the output type is unsupported.
    """
    if output not in ("LogJacs", "thick"):
        raise ValueError("Can't handle output ``{output}``")

    df = pl.read_csv(filename)
    name2cols = _struct_subset2cols(df, struct_subset, output=output)

    df = df.with_columns(
        pl.col("SubjID").str.extract(r"sub-([^_]+)", 1).alias("subj_id"),
        pl.col("SubjID").str.extract(r"_ses-(.+)$", 1).alias("ses_id"),
    )

    if subject_subset is not None:
        df = df.filter(pl.col("subj_id").is_in(subject_subset))

    if session_subset is not None:
        df = df.filter(pl.col("ses_id").is_in(session_subset))

    data = {}
    for row in df.iter_rows(named=True):
        subj_id = row["subj_id"]
        ses_id = row["ses_id"]

        data.setdefault(subj_id, {})[ses_id] = {
            struct_name: np.asarray([row[c] for c in cols])
            for struct_name, cols in name2cols.items()
        }

    return data
