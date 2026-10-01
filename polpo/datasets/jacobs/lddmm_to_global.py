import logging

from polpo.dataset import NestedKeyMap
from polpo.datasets.jacobs.mesh import load_dataset
from polpo.neuroimaging.naming import (
    get_all_subcortical_structs,
    get_subcortical_struct_long_name,
)

from .tabular import get_session_to_week
from .utils import is_pregnancy_subject


def _select_atlas_inputs(dataset, session_to_week):
    """Select observations used to estimate subject-specific atlases.

    For pregnancy subjects, retain only the last available session when it
    occurs after gestational week 42. For non-pregnancy subjects, retain all
    available observations.

    Parameters
    ----------
    dataset : NestedDataset
        Dataset indexed by subject and session.
    session_to_week : NestedKeyMap
        Mapping from subject-session keys to gestational week.

    Returns
    -------
    atlas_inputs : NestedDataset
        Observations to use for atlas estimation.
    """
    last_session = {
        subject: sessions[-1] for subject, sessions in dataset.nested_keys().items()
    }

    def use_for_atlas(subject, session):
        if not is_pregnancy_subject(subject):
            return True

        return (
            session == last_session[subject]
            and session_to_week.map_inner(subject, session) > 42.0
        )

    return dataset.filter_keys(use_for_atlas)


def prepare_inputs(
    struct,
    subject_ids,
    data_dir,
    derivative="enigma",
):
    """Prepare mesh data for an LDDMM-to-global run.

    Load meshes for one subcortical structure, select observations used for
    subject-specific atlas estimation, discard subjects without suitable atlas
    inputs, select observations in the target gestational window, and encode
    dataset keys for use in output paths.

    Parameters
    ----------
    struct : str
        Subcortical structure to process.
    subject_ids : array-like
        Subject identifiers to include.
    data_dir : path-like
        Root directory containing the dataset.
    derivative : str
        Mesh derivative to load.

    Returns
    -------
    dataset : NestedDataset
        Mesh dataset with encoded subject-session keys.
    atlas_keys : mapping
        Encoded subject-session keys identifying observations used for atlas
        estimation.
    atlas_only_keys : mapping
        Encoded subject-session keys used for atlas estimation but excluded
        from the global representations.
    known_correspondences : bool
        Whether the loaded meshes have known vertex correspondences.
    metadata : dict
        Metadata describing the prepared inputs, including the original-to-
        encoded key mapping.
    """
    metadata = dict(
        struct=struct,
        subject_ids=subject_ids,
        derivative=derivative,
        data_dir=data_dir.as_posix(),
    )

    dataset = load_dataset(
        data_dir=data_dir,
        subject_subset=subject_ids,
        struct_subset=[struct],
        derivative=derivative,
        as_surface=True,
    )[struct]

    session_to_week = NestedKeyMap.from_inner_key_map(get_session_to_week())

    atlas_inputs = _select_atlas_inputs(dataset, session_to_week)
    atlas_only_inputs = atlas_inputs.filter_keys(
        lambda subject, _: is_pregnancy_subject(subject)
    )

    # ignores subjects with no atlas keys
    missing_atlas = set(dataset.keys_list()) - set(atlas_inputs.keys_list())
    if missing_atlas:
        logging.info(
            f"Dropping subjects {missing_atlas} because they do not have suitable atlas inputs"
        )
    dataset = dataset.drop_outer(missing_atlas)

    # keeps only pregnancy
    dataset_ = (
        dataset.filter_keys(lambda subject, _: is_pregnancy_subject(subject))
        .map_keys(session_to_week)
        .filter_keys(lambda _, time: 0 <= time <= 42)
        .map_keys(session_to_week.invert())
    )
    dataset_ = dataset_.merge(dataset.select_inner(atlas_inputs))

    # encode dataset keys for manageable folder names
    key_codec = NestedKeyMap.from_dataset(dataset_)

    metadata["key_map"] = key_codec.to_dict()

    known_correspondences = True if derivative == "enigma" else False
    return (
        dataset_.map_keys(key_codec.map),
        key_codec.map_keys(atlas_inputs.nested_keys()),
        key_codec.map_keys(atlas_only_inputs.nested_keys()),
        known_correspondences,
        metadata,
    )


def find_experiment_dirs(outputs_dir, long_name=False, interleave=True):
    """Find subcortical experiment directories.

    Parameters
    ----------
    outputs_dir : path-like
        Directory containing structure-specific experiment directories.
    long_name : bool
        Whether to use long structure names as dictionary keys.
    interleave : bool
        Whether to order left and right hemisphere structures in an
        interleaved sequence.

    Returns
    -------
    experiment_dirs : dict
        Mapping from structure names to experiment directories, ordered
        according to the requested structure ordering.
    """
    all_structs = get_all_subcortical_structs(interleave=interleave)
    order = {struct: i for i, struct in enumerate(all_structs)}

    dirs = sorted(
        (
            path
            for path in outputs_dir.iterdir()
            if path.is_dir() and path.name in order
        ),
        key=lambda path: order[path.name],
    )

    structs = (
        [get_subcortical_struct_long_name(dir_.name) for dir_ in dirs]
        if long_name
        else [dir_.name for dir_ in dirs]
    )

    return dict(zip(structs, dirs))
