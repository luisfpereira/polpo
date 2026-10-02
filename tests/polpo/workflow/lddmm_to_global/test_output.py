from pathlib import Path

import numpy as np
import pytest

from polpo.workflow.lddmm_to_global.output import (
    LddmmToGlobalMultiOutput,
    LddmmToGlobalOutput,
)


@pytest.fixture(scope="session")
def lddmm_output():
    lddmm_output_dir = (Path("tests") / ".test_data" / "lddmm").resolve()

    if not lddmm_output_dir.exists():
        pytest.fail(
            "LDDMM test output is missing. "
            "Generate it with `python -m tests.generate_lddmm_output`."
        )

    return LddmmToGlobalOutput(lddmm_output_dir)


@pytest.mark.smoke
def test_output(lddmm_output):
    output = lddmm_output

    output.global_atlas
    output.global_atlas_point

    # checks all meshes were stored in the outputs folder
    output.dataset.map_values(lambda surface: surface.as_vtk_path())

    output.local_registrations
    output.local_reconstructed_points
    output.registrations_to_global_atlas
    output.global_shoots
    output.global_points
    output.global_deltas
    output.transports
    output.local_atlases
    output.local_atlases_points
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
