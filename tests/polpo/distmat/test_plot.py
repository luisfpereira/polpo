import numpy as np
import pytest
from matplotlib import pyplot as plt

from polpo.distmat.pairwise import PairwiseDistances
from polpo.distmat.plot import plot_dist_comparison, plot_distmat


@pytest.mark.smoke
def test_plot_distmat():
    dists = PairwiseDistances(["a", "b", "c"], np.array([1, 2, 3]))

    ax = plot_distmat(dists, show_ticks=False, title="Distances")

    assert len(ax.images) == 1
    assert ax.get_title() == "Distances"
    plt.close(ax.figure)


@pytest.mark.smoke
def test_plot_dist_comparison():
    labels = ["a", "b", "c"]
    xdist = PairwiseDistances(labels, np.array([1, 2, 3]))
    ydist = PairwiseDistances(labels, np.array([2, 3, 4]))

    ax = plot_dist_comparison(xdist, ydist, s=20)

    assert len(ax.collections) == 1
    assert len(ax.lines) == 1
    np.testing.assert_array_equal(ax.collections[0].get_sizes(), [20])
    plt.close(ax.figure)

    ax = plot_dist_comparison(
        xdist,
        ydist,
        group_by=lambda a, b: a,
        colors={"a": "red", "b": "blue"},
    )

    assert len(ax.collections) == 2
    assert ax.get_legend() is not None
    plt.close(ax.figure)


def test_plot_dist_comparison_invalid_labels():
    xdist = PairwiseDistances(["a", "b"], np.array([1]))
    ydist = PairwiseDistances(["b", "a"], np.array([1]))

    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="Not same key order"):
        plot_dist_comparison(xdist, ydist, ax=ax)

    plt.close(fig)
