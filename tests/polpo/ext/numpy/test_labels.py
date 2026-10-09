import numpy as np
import pytest

from polpo.ext.numpy.labels import select_labels, split_labels


@pytest.fixture
def array():
    return np.array([[0, 1, 2], [3, 1, 0]])


@pytest.mark.parametrize(
    "binary, expected",
    [
        (True, [[False, True, False], [True, True, False]]),
        (False, [[0, 1, 0], [3, 1, 0]]),
    ],
)
def test_select_labels(array, binary, expected):
    """Test binary masks and preservation of selected label values."""
    result = select_labels(array, [1, 3], binary=binary)

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == (bool if binary else array.dtype)


@pytest.mark.parametrize("labels", [[], [99]])
@pytest.mark.parametrize("binary", [True, False])
def test_select_labels_no_matches(array, labels, binary):
    result = select_labels(array, labels, binary=binary)

    expected = np.zeros(array.shape, dtype=bool if binary else array.dtype)
    np.testing.assert_array_equal(result, expected)


def test_select_labels_encoding(array):
    encoding = {"one": 1, "three": 3}

    result = select_labels(
        array,
        ["one", "three"],
        encoding=encoding.__getitem__,
    )

    np.testing.assert_array_equal(result, np.isin(array, [1, 3]))


@pytest.mark.parametrize("binary", [True, False])
def test_split_labels(array, binary):
    result = split_labels(array, [1, 3], binary=binary)

    assert set(result) == {1, 3}

    for label, selected in result.items():
        expected = select_labels(array, [label], binary=binary)
        np.testing.assert_array_equal(selected, expected)


@pytest.mark.parametrize("binary", [True, False])
def test_split_labels_encoding(array, binary):
    encoding = {"one": 1, "three": 3}

    result = split_labels(
        array,
        ["one", "three"],
        binary=binary,
        encoding=encoding.__getitem__,
    )

    assert set(result) == set(encoding)

    for name, label in encoding.items():
        expected = select_labels(array, [label], binary=binary)
        np.testing.assert_array_equal(result[name], expected)


def test_split_labels_empty(array):
    assert split_labels(array, []) == {}
