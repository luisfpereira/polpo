import numpy as np
import pytest

from polpo.distmat import PairDistances


class TestPairDistances:
    def test_pair_distances(self):
        distances = PairDistances(
            [("a", "b"), ("b", "c"), ("a", "d")],
            np.array([1, 2, 3]),
        )

        assert distances.labels == ["a", "b", "c", "d"]
        assert distances.get("a", "b") == 1
        assert distances.get("b", "a") == 1
        assert distances.get("a", "d") == 3
        assert list(distances.items()) == list(zip(distances.pairs, distances.data))
        assert distances.as_dataset()["a", "b"] == 1

    def test_select(self):
        distances = PairDistances(
            [("a", "b"), ("b", "c"), ("a", "d")],
            np.array([1, 2, 3]),
        )

        selected = distances.select(["a", "b", "d"])

        assert selected.pairs == [("a", "b"), ("a", "d")]
        np.testing.assert_array_equal(selected.data, [1, 3])

    def test_select_pairs_and_group_pairs(self):
        distances = PairDistances(
            [("a", "b"), ("a", "c"), ("b", "c")],
            np.array([1, 2, 3]),
        )

        selected = distances.select_pairs([("c", "b"), ("a", "b")])
        np.testing.assert_array_equal(selected.data, [3, 1])

        groups = distances.group_pairs(lambda a, b: a)

        assert set(groups) == {"a", "b"}
        np.testing.assert_array_equal(groups["a"].data, [1, 2])
        np.testing.assert_array_equal(groups["b"].data, [3])

    def test_invalid_length(self):
        with pytest.raises(ValueError):
            PairDistances([("a", "b")], np.array([1, 2]))
