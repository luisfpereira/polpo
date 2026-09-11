import re

import polpo.preprocessing.dict as ppdict
from polpo.bids import DerSessionFolderSelector
from polpo.preprocessing import (
    ExceptionToWarning,
)
from polpo.preprocessing.str import DigitFinder


def _session_sorter(session_id):
    return (
        re.sub(r"\d+$", "", session_id),
        DigitFinder(index=-1)(session_id),
    )


def FoldersSelector(
    derivative,
    subject_subset=None,
    session_subset=None,
    remove_repeated=True,
):
    pipe = DerSessionFolderSelector(
        derivative,
        subject_subset,
        session_subset,
        session_sorter=_session_sorter,
    )

    if remove_repeated:
        # same session metadata as 26
        pipe += ppdict.DictMap(
            ExceptionToWarning(ppdict.RemoveKeys(keys=["27"]), warn=False)
        )

    return pipe
