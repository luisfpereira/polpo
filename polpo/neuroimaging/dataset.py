from polpo.dataset import Dataset, NestedDataset


def group_by_structure(dataset):
    """Group a subject-session dataset by anatomical structure.

    Parameters
    ----------
    dataset : NestedDataset
        Dataset indexed by subject and session, with each value containing
        a ``Dataset`` indexed by structure.

    Returns
    -------
    grouped : Dataset
        Dataset indexed by structure. Each value is a ``NestedDataset``
        indexed by subject and session.
    """
    data = {}

    for subject_id, sessions in dataset.items():
        for session_id, structures in sessions.items():
            for struct_id, value in structures.items():
                data.setdefault(struct_id, {}).setdefault(subject_id, {})[
                    session_id
                ] = value

    return Dataset(
        {
            struct_id: NestedDataset(struct_data)
            for struct_id, struct_data in data.items()
        }
    )
