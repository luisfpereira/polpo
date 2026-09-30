import string

import pytest

from polpo.dataset import NestedDataset
from polpo.surface_mesh.core import Surface
from polpo.surface_mesh.generation.blob import create_blob
from polpo.workflow.lddmm_to_global import (
    LddmmToGlobalMultiOutput,
    LddmmToGlobalOutput,
)

try:
    from polpo.workflow.lddmm_to_global import LddmmToGlobal
except ImportError:
    pass


@pytest.fixture(scope="session")
def lddmm_output(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("lddmm")

    data = {}
    for subj_index, (n_meshes, bump_amp, n_bumps) in enumerate(
        zip((3, 2, 4), (0.2, 0.3, 0.4), (3, 5, 6))
    ):
        data[string.ascii_uppercase[subj_index + 3]] = {
            index + 2: Surface.from_polydata(
                create_blob(
                    resolution=10, bump_amp=bump_amp, n_bumps=n_bumps, smoothing_iter=10
                )
            )
            for index in range(n_meshes)
        }

    dataset = NestedDataset(data)

    atlas_keys = {
        "D": [2, 3],
        "E": [2],
        "F": [2, 3],
    }

    protocol = LddmmToGlobal(
        known_correspondences=True,
        results_dir=tmp_path,
    )
    protocol.run(dataset, atlas_keys=atlas_keys)

    return LddmmToGlobalOutput(tmp_path)


@pytest.mark.slow
@pytest.mark.deformetrica
@pytest.mark.smoke
def test_runs(lddmm_output):
    pass


@pytest.mark.slow
@pytest.mark.deformetrica
@pytest.mark.smoke
def test_output_view(lddmm_output):
    view = lddmm_output.view

    view.dataset
    view.local_atlases
    view.local_registrations
    view.local_reconstructed_points
    view.registrations_to_global_atlas
    view.global_shoots
    view.global_points
    view.global_deltas
    view.transports
    view.global_atlas
    view.global_atlas_point
    view.global_atlas_flows


@pytest.mark.slow
@pytest.mark.deformetrica
@pytest.mark.smoke
def test_multi_output_view(lddmm_output):
    lddmm_multi_output = LddmmToGlobalMultiOutput([lddmm_output, lddmm_output])

    view = lddmm_multi_output.view

    view.dataset
    view.local_reconstructed_points
    view.global_points
