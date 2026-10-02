import numpy as np
import pytest

from polpo.workflow.lddmm_to_global.output import LddmmToGlobalMultiOutput

from .fixtures import lddmm_output  # noqa: F401


@pytest.mark.smoke
def test_output(lddmm_output):
    output = lddmm_output

    output.global_atlas
    output.global_atlas_point

    output.dataset

    output.local_registrations

    output.local_reconstructed_points

    output.registrations_to_global_atlas

    output.global_shoots
    output.global_points

    output.global_deltas

    output.transports

    output.local_atlases
    output.local_atlases_points
    output.local_atlases.map_values(lambda res: res.points)
    output.local_atlases.map_values(lambda res: res.reconstructed)

    output.global_atlas_flows


@pytest.mark.smoke
def test_multi_output(lddmm_output):
    output = LddmmToGlobalMultiOutput([lddmm_output, lddmm_output])

    output.local_reconstructed_points
    output.global_points

    outer = "D"
    inner = 2

    np.testing.assert_allclose(
        2 * lddmm_output.dataset[outer][inner].as_surface().to_polydata().volume,
        output.dataset[outer][inner].as_surface().to_polydata().volume,
    )
