from .tabular import get_session_to_week  # noqa: F401

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
    """Get maternal project subject identifiers.

    Parameters
    ----------
    include_pilot : bool
        Whether to include the pilot subject.
    include_male : bool
        Whether to include male subjects.
    include_control : bool
        Whether to include control subjects.
    sort : bool
        Whether to sort the returned identifiers.

    Returns
    -------
    ids : list of str
        Subject identifiers satisfying the requested filters.
    """
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


def validate_subject_subset(subject_subset):
    """Validate subject identifiers.

    Parameters
    ----------
    subject_subset : array-like
        Subject identifiers to validate.

    Raises
    ------
    ValueError
        If any subject identifier is not available.
    """
    invalid = set(subject_subset) - set(MATERNAL_IDS)
    if invalid:
        raise ValueError(
            f"Unknown subject IDs: {sorted(invalid)}. "
            f"Available IDs: {sorted(MATERNAL_IDS)}."
        )
