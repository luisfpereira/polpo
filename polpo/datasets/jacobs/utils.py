import polpo.preprocessing.dict as ppdict
from polpo.preprocessing import IdentityStep

from .tabular import get_key_to_birth_week, get_key_to_week

MATERNAL_IDS = {
    "01",
    "1001B",
    "1003B",
    "1004B",
    "1009B",
    "1011B",
    "1012B",
    "2003B",
    "2004B",
    "2011B",
    "2012B",
    "3003B",
    "3004B",
    "4001B",
}


def get_subject_ids(
    include_pilot=True, include_male=True, include_control=True, sort=False
):
    ids = MATERNAL_IDS.copy()

    if not include_pilot:
        ids.remove("01")

    for id_ in ids.copy():
        if not include_male and id_.startswith("2"):
            ids.remove(id_)

        if not include_control and (id_.startswith("3") or id_.startswith("4")):
            ids.remove(id_)

    if sort:
        ids = sorted(ids)

    return ids


def _index_session_by_step(index_session_by="id", data_dir=None, subject_subset=None):
    if index_session_by not in ("id", "gest_week", "birth"):
        raise ValueError("Can't handle indexing by ``{index_session_by}``")

    if index_session_by == "gest_week":
        keys_to_weeks = get_key_to_week(data_dir, subject_subset=subject_subset)
        return ppdict.RenameNestedKeys(keys_to_weeks)

    if index_session_by == "birth":
        keys_to_weeks = get_key_to_birth_week(data_dir, subject_subset=subject_subset)
        return ppdict.RenameNestedKeys(keys_to_weeks)

    return IdentityStep()
