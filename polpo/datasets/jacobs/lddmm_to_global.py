import logging

import polpo.preprocessing.dict as ppdict
from polpo.dataset import Dataset, NestedDataset, NestedKeyMap
from polpo.jacobs.mesh import MeshDatasetLoader
from polpo.jacobs.tabular import get_key_to_week
from polpo.neuroi.naming import (
    get_all_subcortical_structs,
    get_subcortical_struct_long_name,
)


def prepare_inputs(
    struct,
    subject_ids,
    data_dir,
    derivative="enigma",
):
    """Prepare Jacobs data for an LDDMM-to-global run.

    Load meshes for a single subcortical structure, select the observations
    used as local atlases, remove subjects without a suitable atlas, and
    encode dataset keys for use in output paths.

    Parameters
    ----------
    struct : str
        Subcortical structure to process.
    subject_ids : array-like
        Subject identifiers to include.
    data_dir : path-like
        Root directory containing the Jacobs dataset.
    derivative : str
        Mesh derivative to load.

    Returns
    -------
    dataset : NestedDataset
        Mesh dataset with encoded keys.
    atlas_keys : NestedDataset
        Keys identifying the observations used as local atlases.
    metadata : dict
        Metadata describing the prepared dataset, including the key mapping
        and atlas keys.
    known_correspondences : bool
        Whether the loaded meshes have known vertex correspondences.
    """
    metadata = dict(
        struct=struct,
        subject_ids=subject_ids,
        derivative=derivative,
        data_dir=data_dir,
    )

    dataset = (
        MeshDatasetLoader(
            data_dir=data_dir,
            subject_subset=subject_ids,
            struct_subset=[struct],
            derivative=derivative,
            mesh_reader=None,
        )
        + ppdict.ExtractUniqueKey(nested=True)
        + NestedDataset
    )()

    last_keys = Dataset(dataset.inner_keys()).map_values(lambda x: x[-1])

    def local_template_filter(subj, session):
        control = int(subj[0]) > 1
        if control:
            return True
        else:
            return last_keys[subj] == session

    # ignores subjects with no atlas keys
    atlas_keys = dataset.filter_keys(local_template_filter)

    missing_pre = set(dataset.keys_list()) - set(atlas_keys.keys_list())
    if missing_pre:
        logging.info(
            f"Dropping subjects {missing_pre} because they do not have baseline meshes"
        )
    dataset = dataset.drop_outer(missing_pre)

    # encode dataset keys for manageable folder names
    key_codec = NestedKeyMap.from_dataset(dataset)

    metadata["key_map"] = key_codec.to_dict()
    mapped_atlas_keys = metadata["atlas_keys"] = key_codec.encode_nested_keys(
        atlas_keys.inner_keys()
    )

    known_correspondences = True if derivative == "enigma" else False
    return (
        dataset.map_keys(key_codec.encode),
        mapped_atlas_keys,
        known_correspondences,
        metadata,
    )


def find_experiment_dirs(outputs_dir, long_name=False, interleave=True):
    all_structs = get_all_subcortical_structs(interleave=interleave)
    order = {struct: i for i, struct in enumerate(all_structs)}

    dirs = sorted(
        [file for file in outputs_dir.iterdir() if file.is_dir() if file.name in order],
        key=lambda file: order[file.name],
    )

    all_structs = get_all_subcortical_structs(interleave=False)

    dirs = sorted(
        [
            file
            for file in outputs_dir.iterdir()
            if file.is_dir()
            if file.name in all_structs
        ],
        key=lambda file: order[file.name],
    )
    structs = (
        [get_subcortical_struct_long_name(dir_.name) for dir_ in dirs]
        if long_name
        else [dir_.name for dir_ in dirs]
    )

    return dict(zip(structs, dirs))


def get_output_week_view(source):
    key2week_codec = NestedKeyMap.from_inner_key_map(get_key_to_week())
    return source.decoded.with_key_map(key2week_codec)
