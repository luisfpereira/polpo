import pytest

from polpo.neuroimaging.fsl.mesh import select_mesh_paths


@pytest.fixture
def fsl_mesh_dir(tmp_path):
    filenames = [
        "sub-AAA_ses-visit1-L_Hipp_first.vtk",
        "sub-AAA_ses-visit1-R_Hipp_first.vtk",
        "sub-AAA_ses-visit1-L_Thal_first.vtk",
        # distractors
        "sub-AAA_ses-visit1-L_Hipp_first.bvars",
        "sub-AAA_ses-visit1_first_stats.csv",
    ]

    for filename in filenames:
        (tmp_path / filename).touch()

    return tmp_path


def test_select_mesh_paths(fsl_mesh_dir):
    struct_subset = ["R_Hipp", "L_Thal", "L_Hipp"]

    paths = select_mesh_paths(
        fsl_mesh_dir,
        struct_subset=struct_subset,
    )

    assert paths.keys_list() == struct_subset

    assert paths["R_Hipp"] == (fsl_mesh_dir / "sub-AAA_ses-visit1-R_Hipp_first.vtk")
    assert paths["L_Thal"] == (fsl_mesh_dir / "sub-AAA_ses-visit1-L_Thal_first.vtk")
    assert paths["L_Hipp"] == (fsl_mesh_dir / "sub-AAA_ses-visit1-L_Hipp_first.vtk")
