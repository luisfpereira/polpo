import numpy as np
import pytest
from matplotlib import pyplot as plt

from polpo.distmat import PairwiseDistances
from polpo.distmat.pairwise import pairwise_dists


def test_pairwise_dists():
    points = [0, 1, 3, 7]
    dist_fnc = lambda a, b: abs(a - b)

    expected = np.array([1, 3, 7, 2, 6, 4])
    expected_matrix = np.array(
        [
            [0, 1, 3, 7],
            [1, 0, 2, 6],
            [3, 2, 0, 4],
            [7, 6, 4, 0],
        ]
    )

    np.testing.assert_array_equal(
        pairwise_dists(points, dist_fnc, as_matrix=False), expected
    )
    np.testing.assert_array_equal(pairwise_dists(points, dist_fnc), expected_matrix)


def test_pairwise_dists_parallel():
    points = [0, 1, 3, 7]
    dist_fnc = lambda a, b: abs(a - b)

    sequential = pairwise_dists(points, dist_fnc, as_matrix=False, n_jobs=1)
    parallel = pairwise_dists(points, dist_fnc, as_matrix=False, n_jobs=2)

    np.testing.assert_array_equal(parallel, sequential)


class TestPairwiseDistances:
    def test_matrix_and_get(self):
        distances = PairwiseDistances(["a", "b", "c"], np.array([1, 4, 2]))

        np.testing.assert_array_equal(
            distances.matrix,
            [[0, 1, 4], [1, 0, 2], [4, 2, 0]],
        )
        assert distances.pairs == [("a", "b"), ("a", "c"), ("b", "c")]
        assert distances.get("c", "a") == 4
        assert distances.get("a", "a") == 0

        reconstructed = PairwiseDistances.from_matrix(
            distances.labels, distances.matrix
        )
        np.testing.assert_array_equal(reconstructed.data, distances.data)

    def test_invalid_data_length(self):
        with pytest.raises(ValueError):
            PairwiseDistances(["a", "b", "c"], np.array([1, 2]))

    def test_select_and_map_labels(self):
        distances = PairwiseDistances(["a", "b", "c"], np.array([1, 4, 2]))

        selected = distances.select(["c", "a"])
        assert selected.labels == ["c", "a"]
        np.testing.assert_array_equal(selected.data, [4])

        filtered = distances.filter_labels(lambda label: label != "b")
        assert filtered.labels == ["a", "c"]
        np.testing.assert_array_equal(filtered.data, [4])

        mapped = distances.map_labels(str.upper)
        assert mapped.labels == ["A", "B", "C"]
        np.testing.assert_array_equal(mapped.data, distances.data)

    @pytest.mark.parametrize(
        "descending, expected",
        [(True, ["c", "a", "b"]), (False, ["b", "a", "c"])],
    )
    def test_sort_by_row_norm(self, descending, expected):
        distances = PairwiseDistances(["a", "b", "c"], np.array([1, 4, 2]))

        result = distances.sort_by_row_norm(descending=descending)

        assert result.labels == expected
        np.testing.assert_array_equal(result.data, distances.select(expected).data)

    def test_merge_many(self):
        first = PairwiseDistances(["a", "b"], np.array([1]))
        second = PairwiseDistances(["c", "d"], np.array([2]))

        merged = PairwiseDistances.merge_many([first, second])

        assert merged.pairs == [("a", "b"), ("c", "d")]
        np.testing.assert_array_equal(merged.data, [1, 2])

    def test_save_load(self, tmp_path):
        distances = PairwiseDistances(["a", "b", "c"], np.array([1, 4, 2]))
        path = tmp_path / "distances.npz"

        distances.save(path)
        loaded = PairwiseDistances.load(path)

        assert loaded.labels == distances.labels
        np.testing.assert_array_equal(loaded.data, distances.data)

    @pytest.mark.smoke
    def test_plot(self):
        distances = PairwiseDistances(["a", "b", "c"], np.array([1, 4, 2]))

        ax = distances.plot()

        assert len(ax.images) == 1
        plt.close(ax.figure)
