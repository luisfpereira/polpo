from pathlib import Path

import polpo.preprocessing.dict as ppdict
from polpo.bids import DerFolderSelector
from polpo.enigma.output import load_output
from polpo.preprocessing import BranchingPipeline

from .defaults import PILOT_PROJECT_FOLDER, PROJECT_FOLDER
from .path import _session_sorter, _split_subject_subset
from .utils import _index_session_by_step


def OutputLoader(
    subject_subset=None,
    session_subset=None,
    struct_subset=None,
    index_session_by="id",
    remove_repeated=True,
    output="LogJacs",
):
    """Log jacobian loader.

    Parameters
    ----------
    subject_subset : array-like
        Subject identifiers to select. If ``None``, all subjects are used.
    session_subset : array-like
        Session identifiers to select. If ``None``, all sessions are used.
    struct_subset : array-like
        Structure identifiers to select. If ``None``, all structures are used.
    index_session_by : {"id", "gest_week", "birth"}
        Strategy used to index sessions in the output.

        - ``"id"``: keep the original session identifiers.
        - ``"gest_week"``: replace session identifiers with gestational
        weeks.
        - ``"birth"``: replace session identifiers with gestational weeks
        relative to birth (birth week corresponds to 0).
    output : {"LogJacs", "thick"}
    """
    pilot_subset, subject_subset_ = _split_subject_subset(subject_subset)
    derivative = "enigma"

    pipes = []
    if len(pilot_subset):
        pipe = (
            (lambda folder: Path(folder).expanduser() / PILOT_PROJECT_FOLDER)
            + DerFolderSelector(derivative)
            + (lambda path: path / "data" / f"subjects_file_{output}.csv")
            + (
                lambda filename: load_output(
                    filename,
                    subject_subset=pilot_subset,
                    session_subset=session_subset,
                    struct_subset=struct_subset,
                    output=output,
                )
            )
            + (ppdict.DictMap(ppdict.RemoveKeys(["27"])) if remove_repeated else None)
        )
        pipes.append(pipe)

    if len(subject_subset_):
        pipe = (
            (lambda folder: Path(folder).expanduser() / PROJECT_FOLDER)
            + DerFolderSelector(derivative)
            + (lambda path: path / "data" / f"subjects_file_{output}.csv")
            + (
                lambda filename: load_output(
                    filename,
                    subject_subset=subject_subset_,
                    session_subset=session_subset,
                    struct_subset=struct_subset,
                    output=output,
                )
            )
            + ppdict.DictMap(ppdict.KeySorter(_session_sorter))
        )
        pipes.append(pipe)

    # TODO: handle data_dir consistently
    index_session_step = _index_session_by_step(
        index_session_by,
        subject_subset=subject_subset,
    )

    if len(pipes) == 1:
        return pipes[0] + index_session_step

    return (
        BranchingPipeline(pipes, merger=lambda data: data[0] | data[1])
        + index_session_step
    )


def LogJacsLoader(
    subject_subset=None,
    session_subset=None,
    struct_subset=None,
    index_session_by="id",
    remove_repeated=True,
):
    return OutputLoader(
        subject_subset=subject_subset,
        session_subset=session_subset,
        struct_subset=struct_subset,
        index_session_by=index_session_by,
        remove_repeated=remove_repeated,
    )
