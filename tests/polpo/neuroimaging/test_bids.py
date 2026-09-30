import pytest

from polpo.neuroimaging.bids import select_folders


@pytest.fixture
def bids_dir(tmp_path):
    for subject_id in ["AAA", "BBB"]:
        for session_id in ["time1", "time2"]:
            (tmp_path / f"sub-{subject_id}_ses-{session_id}").mkdir()

    return tmp_path


def test_select_folders(bids_dir):
    folders = select_folders(bids_dir).sort_keys().sort_inner_keys()

    assert folders.nested_keys() == {
        "AAA": ["time1", "time2"],
        "BBB": ["time1", "time2"],
    }


@pytest.mark.parametrize("subject_id", ["AAA", "BBB"])
@pytest.mark.parametrize("session_id", ["time1", "time2"])
def test_select_folders_subset(bids_dir, subject_id, session_id):
    folders = select_folders(
        bids_dir,
        subject_subset=[subject_id],
        session_subset=[session_id],
    ).sort_inner_keys()

    assert folders.nested_keys() == {
        subject_id: [session_id],
    }
