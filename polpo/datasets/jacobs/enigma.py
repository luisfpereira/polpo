from pathlib import Path

from polpo.dataset import NestedDataset
from polpo.neuroimaging.bids import find_derivative_dir
from polpo.neuroimaging.enigma.output import load_output as load_enigma_output

from .defaults import DATA_DIR, MATERNAL_PROJECT_FOLDER, PILOT_PROJECT_FOLDER
from .maternal.path import _session_sort_key
from .path import _split_subject_subset


def load_output(
    data_dir=None,
    subject_subset=None,
    session_subset=None,
    struct_subset=None,
    output="LogJacs",
    remove_repeated=True,
):
    """Load ENIGMA outputs indexed by subject and session.

    Parameters
    ----------
    data_dir : path-like
        Dataset root directory.
    subject_subset : array-like
        Subject identifiers to select. If ``None``, all subjects are used.
    session_subset : array-like
        Session identifiers to select. If ``None``, all sessions are used.
    struct_subset : array-like
        Structure identifiers to select. If ``None``, all structures are used.
    output : str
        ENIGMA output type, either ``"LogJacs"`` or ``"thick"``.
    remove_repeated : bool
        Whether to exclude repeated pilot session ``27``.

    Returns
    -------
    dataset : NestedDataset
        ENIGMA outputs indexed by subject and session.
        Each value maps structure identifiers to arrays.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    data_dir = Path(data_dir).expanduser()
    pilot_subset, maternal_subset = _split_subject_subset(subject_subset)

    datasets = []

    for folder, subjects, sort_key in (
        (PILOT_PROJECT_FOLDER, pilot_subset, int),
        (MATERNAL_PROJECT_FOLDER, maternal_subset, _session_sort_key),
    ):
        if not subjects:
            continue

        derivative_dir = find_derivative_dir(data_dir / folder, "enigma")
        filename = derivative_dir / "data" / f"subjects_file_{output}.csv"

        dataset = NestedDataset(
            load_enigma_output(
                filename,
                subject_subset=subjects,
                session_subset=session_subset,
                struct_subset=struct_subset,
                output=output,
            )
        ).sort_inner_keys(sort_key)

        if folder == PILOT_PROJECT_FOLDER and remove_repeated:
            dataset = dataset.drop_inner({"01": ["27"]})

        datasets.append(dataset)

    return NestedDataset.merge_many(datasets).sort_keys()


def load_log_jacs(
    data_dir=None,
    subject_subset=None,
    session_subset=None,
    struct_subset=None,
    remove_repeated=True,
):
    """Load ENIGMA log-Jacobian outputs."""
    return load_output(
        data_dir=data_dir,
        subject_subset=subject_subset,
        session_subset=session_subset,
        struct_subset=struct_subset,
        output="LogJacs",
        remove_repeated=remove_repeated,
    )
