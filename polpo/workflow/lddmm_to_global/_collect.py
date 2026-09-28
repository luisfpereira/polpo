from polpo.dataset import Dataset, NestedDataset
from polpo.surface_mesh.deformetrica.representations import Point
from polpo.surface_mesh.deformetrica.results import (
    DeterministicAtlasResult,
    RegistrationResult,
    ShootResult,
    TransportResult,
)


def collect_dataset(dir_config, dataset_keys):
    return NestedDataset.from_keys(
        dataset_keys,
        lambda outer_key, inner_key: Point(
            id_=f"{outer_key}-{inner_key}",
            dirname=dir_config.meshes,
        ),
    )


def collect_local_registrations(dir_config, dataset_keys):
    return NestedDataset.from_keys(
        dataset_keys,
        lambda outer_key, inner_key: RegistrationResult.load(
            f"{outer_key}_to_{outer_key}-{inner_key}",
            dir_config,
        ),
    )


def collect_global_shoots(dir_config, dataset_keys, atlas_id="gl"):
    return NestedDataset.from_keys(
        dataset_keys,
        lambda outer_key, inner_key: ShootResult.load(
            f"{atlas_id}_shoot_{outer_key}_to_{outer_key}-{inner_key}"
            f"_along_{outer_key}_to_{atlas_id}",
            dir_config,
        ),
    )


def collect_atlases(dir_config, dataset_keys):
    return Dataset.from_keys(
        dataset_keys,
        lambda key: DeterministicAtlasResult.load(key, dir_config),
    )


def get_global_atlas(dir_config, atlas_id="gl"):
    return DeterministicAtlasResult.load(atlas_id, dir_config)


def collect_transports(dir_config, dataset_keys, atlas_id="gl"):
    return NestedDataset.from_keys(
        dataset_keys,
        lambda outer_key, inner_key: TransportResult.load(
            f"{outer_key}_to_{outer_key}-{inner_key}"
            f"_along_{outer_key}_to_{atlas_id}",
            dir_config,
        ),
    )


def collect_registrations_to_global_atlas(
    dir_config,
    outer_keys,
    atlas_id="gl",
):
    return Dataset.from_keys(
        outer_keys,
        lambda outer_key: RegistrationResult.load(
            f"{outer_key}_to_{atlas_id}",
            dir_config,
        ),
    )
