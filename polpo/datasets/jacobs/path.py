import re
from pathlib import Path

from polpo.bids import DerSessionFolderSelector
from polpo.preprocessing import BranchingPipeline
from polpo.preprocessing.str import DigitFinder

from .defaults import PILOT_PROJECT_FOLDER, PROJECT_FOLDER
from .pilot.path import FoldersSelector as PilotFoldersSelector
from .utils import MATERNAL_IDS


def _session_sorter(session_id):
    return (
        re.sub(r"\d+$", "", session_id),
        DigitFinder(index=-1)(session_id),
    )


def _split_subject_subset(subject_subset=None):
    if subject_subset is None:
        subject_subset = MATERNAL_IDS

    for subject_id in subject_subset:
        if subject_id not in MATERNAL_IDS:
            raise ValueError(
                f"Oops, `{subject_id}` is not available. Please, choose from: {','.join(MATERNAL_IDS)}"
            )

    if "01" in subject_subset:
        subject_subsets = [{"01"}]
        subject_subset = subject_subset = list(subject_subset)
        subject_subset.remove("01")
    else:
        subject_subsets = [{}]

    subject_subsets.append(subject_subset)

    return subject_subsets


def _raise_session_subject(subject_subset, session_subset):
    if (
        session_subset is not None
        and len(subject_subset) > 1
        and "01" in subject_subset
    ):
        raise ValueError("Can't filter sessions if pilot included")


def FoldersSelector(
    derivative,
    subject_subset=None,
    session_subset=None,
    remove_repeated=True,
):
    """Create pipeline to select derivative session folders.

    The pipeline takes a dataset root directory as input and returns
    a nested dictionary indexed by subject and session identifiers:

    ``output[subject_id][session_id] -> folder_path``

    Assumes the following directory structure:

    - <data_dir>
        - maternal_brain_project
        - maternal_brain_project_pilot

    For each project folder:

    - <project_folder>
        - derivatives
            - <derivative_1>
            - <derivative_2>

    For each derivative folder:

    - <derivative>
        - <session-folder>
        - ...

    Parameters
    ----------
    derivative : str
        Name of the derivative folder (e.g. ``"fsl_first"``,
        ``"fastsurfer-long"``).
    subject_subset : array-like, optional
        Subject identifiers to select. If ``None``, all subjects are used.
    session_subset : array-like, optional
        Session identifiers to select. If ``None``, all sessions are used.
    remove_repeated : bool, optional
        Whether to remove repeated subject-session entries across projects.

    Returns
    -------
    pipe : Pipeline
        Pipeline mapping a dataset root directory to a nested dictionary
        of derivative session folder paths indexed by subject and session.
    """
    if subject_subset is None:
        subject_subset = sorted(MATERNAL_IDS)

    _raise_session_subject(subject_subset, session_subset)

    pilot_subset, subject_subset = _split_subject_subset(subject_subset)

    pipes = []
    if len(pilot_subset):
        pipe = (lambda path: Path(path) / PILOT_PROJECT_FOLDER) + PilotFoldersSelector(
            derivative,
            subject_subset=pilot_subset,
            session_subset=session_subset,
            remove_repeated=remove_repeated,
        )
        pipes.append(pipe)

    if len(subject_subset):
        excluded_sessions = {
            "1011B": {"pre1", "pre2"},
        }
        pipe = (
            (lambda path: Path(path) / PROJECT_FOLDER)
            + DerSessionFolderSelector(
                derivative,
                subject_subset,
                session_subset,
                session_sorter=_session_sorter,
            )
            + (
                lambda data: {
                    subject: {
                        session: value
                        for session, value in sessions.items()
                        if session not in excluded_sessions.get(subject, ())
                    }
                    for subject, sessions in data.items()
                }
            )
        )
        pipes.append(pipe)

    if len(pipes) == 1:
        return pipes[0]

    return BranchingPipeline(pipes, merger=lambda x: x[0] | x[1])
