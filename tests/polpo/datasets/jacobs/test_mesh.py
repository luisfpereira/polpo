import pytest

from polpo.datasets.jacobs.mesh import load_dataset


@pytest.fixture
def enigma_data_dir(tmp_path):
    derivative_dir = tmp_path / "derivatives" / "enigma_shape"
    derivative_dir.mkdir(parents=True)

    aseg_ids = [10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58, 16]

    sessions = [
        ("1001B", "gest1"),
        ("1001B", "gest2"),
        ("1011B", "pre1"),
        ("1011B", "pre2"),
        ("1011B", "gest1"),
    ]

    folders = {
        "maternal_brain_project_pilot": [
            ("01", "01"),
            ("01", "02"),
            ("01", "26"),
            ("01", "27"),
        ],
        "maternal_brain_project": [
            ("1009B", "pre2"),
            ("1011B", "pre1"),
            ("1011B", "pre2"),
            ("1011B", "gest1"),
            ("1001B", "gest1"),
        ],
    }

    for project, sessions in folders.items():
        derivative_dir = tmp_path / project / "derivatives" / "enigma_shape"

        for subject_id, session_id in sessions:
            session_dir = derivative_dir / f"sub-{subject_id}_ses-{session_id}"
            session_dir.mkdir(parents=True)

            for aseg_id in aseg_ids:
                (session_dir / f"resliced_mesh_{aseg_id}").touch()

    return tmp_path


@pytest.mark.smoke
def test_load_dataset(enigma_data_dir):
    dataset = load_dataset(
        "enigma",
        data_dir=enigma_data_dir,
        subject_subset=["01", "1001B", "1009B", "1011B"],
        struct_subset=["L_Hipp"],
        as_surface=False,
    )

    assert dataset.keys_list() == ["L_Hipp"]

    hipp = dataset["L_Hipp"]

    assert set(hipp.keys()) == {"01", "1001B", "1009B", "1011B"}

    assert set(hipp["01"]) == {"01", "02", "26"}
    assert set(hipp["1001B"]) == {"gest1"}
    assert set(hipp["1009B"]) == {"pre2"}
    assert set(hipp["1011B"]) == {"gest1"}

    assert all(
        path.name == "resliced_mesh_17"
        for sessions in hipp.values()
        for path in sessions.values()
    )
