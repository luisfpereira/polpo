import pytest

from polpo.neuroimaging.enigma.mesh import (
    get_template_vertex_count,
    load_template,
    select_mesh_paths,
)


@pytest.fixture
def enigma_mesh_dir(tmp_path):
    filenames = [
        "resliced_mesh_17",
        "resliced_mesh_53",
        "resliced_mesh_10",
        "resliced_mesh_17.LogJacs",
        "resliced_mesh_17.m",
        "resliced_mesh_17.thick",
        "resliced_mesh_17_to_aseg",
        "LogJacs_17.raw",
        "curve_17.ucf",
    ]

    for filename in filenames:
        (tmp_path / filename).touch()

    return tmp_path


def test_select_mesh_paths(enigma_mesh_dir):
    struct_subset = ["R_Hipp", "L_Hipp", "L_Thal"]

    paths = select_mesh_paths(
        enigma_mesh_dir,
        struct_subset=struct_subset,
    )

    assert paths.keys_list() == struct_subset
    assert paths["R_Hipp"] == enigma_mesh_dir / "resliced_mesh_53"
    assert paths["L_Hipp"] == enigma_mesh_dir / "resliced_mesh_17"
    assert paths["L_Thal"] == enigma_mesh_dir / "resliced_mesh_10"


@pytest.mark.parametrize("hemi", ["L", "R"])
@pytest.mark.parametrize(
    "struct",
    ["Thal", "Caud", "Puta", "Pall", "Hipp", "Amyg", "Accu"],
)
def test_load_template_n_vertices(hemi, struct):
    struct = f"{hemi}_{struct}"

    surface = load_template(struct)

    assert len(surface.vertices) == get_template_vertex_count(struct)
