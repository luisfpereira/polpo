import numpy as np

from polpo.ext.numpy.io import load_indexed_array, save_indexed_array
from polpo.linalg import permute_by_row_norm, sym_to_triu_vec, triu_vec_to_sym

from .pairs import BasePairDistances, PairDistances
from .plot import plot_distmat


def pairwise_dists(points, dist_fnc, as_matrix=True, n_jobs=1):
    """Compute pairwise distances between points.

    Parameters
    ----------
    points : sequence
        Points to compare.
    dist_fnc : callable
        Function taking two points and returning their distance.
    as_matrix : bool
        Whether to return a symmetric distance matrix or a condensed
        vector containing the upper triangle, excluding the diagonal.
    n_jobs : int
        Number of parallel workers. Use 1 for sequential execution
        and -1 to use all available workers.

    Returns
    -------
    dists : ndarray, shape (n, n) or (n * (n - 1) // 2,)
        Pairwise distances, where n is the number of points.
        The diagonal is zero when returned as a matrix.
    """
    if n_jobs == 1:
        dists = [
            dist_fnc(point, cmp_point)
            for index, point in enumerate(points)
            for cmp_point in points[index + 1 :]
        ]
    else:
        dists = _pairwise_dists_par(points, dist_fnc, n_jobs)

    dists = np.array(dists)
    return triu_vec_to_sym(dists) if as_matrix else dists


def _pairwise_dists_par(points, dist_fnc, n_jobs=None):
    """Compute condensed pairwise distances using parallel workers."""
    from joblib import Parallel, delayed

    row_ind, col_ind = np.triu_indices(len(points), k=1)

    return Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(dist_fnc)(points[i], points[j]) for i, j in zip(row_ind, col_ind)
    )


class PairwiseDistances(BasePairDistances):
    """Pairwise distances stored in condensed form.

    Parameters
    ----------
    labels : sequence, shape=(n,)
        Sample labels.
    data : array-like, shape=(n * (n - 1) / 2,)
        Upper-triangular distances in condensed form.
    """

    def __init__(self, labels, data):
        self.labels = labels
        self.data = data

        self._validate()

        self._label_to_index = {label: index for index, label in enumerate(self.labels)}

    def _validate(self):
        expected = len(self.labels) * (len(self.labels) - 1) // 2

        if len(self.data) != expected:
            raise ValueError(f"Expected {expected} distances, got {len(self.data)}.")

    @property
    def matrix(self):
        """Return the symmetric distance matrix.

        Returns
        -------
        matrix : array-like, shape=(n, n)
            Pairwise distance matrix.
        """
        return triu_vec_to_sym(self.data)

    @classmethod
    def from_matrix(cls, labels, matrix):
        """Create pairwise distances from a matrix.

        Parameters
        ----------
        labels : sequence, shape=(n,)
            Sample labels.
        matrix : array-like, shape=(n, n)
            Symmetric pairwise distance matrix.

        Returns
        -------
        distances : PairwiseDistances
            Pairwise distances in condensed form.
        """
        return cls(labels, sym_to_triu_vec(matrix))

    @property
    def pairs(self):
        """Return label pairs in storage order.

        Returns
        -------
        pairs : list, shape=(n * (n - 1) / 2,)
            Pairs corresponding to the stored distances.
        """
        return [
            (self.labels[i], self.labels[j])
            for i in range(len(self.labels))
            for j in range(i + 1, len(self.labels))
        ]

    def _index(self, label):
        """Return the index associated with a label.

        Parameters
        ----------
        label
            Sample label.

        Returns
        -------
        index : int
            Label index.
        """
        return self._label_to_index[label]

    @staticmethod
    def _pair_index(i, j, n):
        """Return the condensed index corresponding to a pair of distinct samples.

        Parameters
        ----------
        i : int
            First sample index.
        j : int
            Second sample index.
        n : int
            Number of samples.

        Returns
        -------
        index : int
            Position of the pair in the condensed upper triangle.
        """
        if i > j:
            i, j = j, i

        return n * i - i * (i + 1) // 2 + (j - i - 1)

    def get(self, label_a, label_b):
        """Return the distance between two labels.

        Parameters
        ----------
        label_a
            First sample label.
        label_b
            Second sample label.

        Returns
        -------
        distance : float
            Pairwise distance.
        """
        i = self._index(label_a)
        j = self._index(label_b)

        if i == j:
            return 0.0

        index = self._pair_index(i, j, len(self.labels))
        return self.data[index]

    def save(self, path):
        """Save pairwise distances.

        Parameters
        ----------
        path : path-like
            Output path.
        """
        return save_indexed_array(path, self.labels, self.data)

    @classmethod
    def load(cls, path):
        """Load pairwise distances.

        Parameters
        ----------
        path : path-like
            Input path.

        Returns
        -------
        distances : PairwiseDistances
            Loaded pairwise distances.
        """
        labels, data = load_indexed_array(path)
        return cls(labels=labels, data=data)

    def sort_by_row_norm(self, descending=True):
        """Sort samples by distance-matrix row norm.

        Parameters
        ----------
        descending : bool
            Whether to order by decreasing row norm.

        Returns
        -------
        distances : PairwiseDistances
            Pairwise distances with reordered labels.
        """
        perm_mat, sorted_idx = permute_by_row_norm(self.matrix, descending=descending)
        labels = [self.labels[index] for index in sorted_idx]
        return self.__class__.from_matrix(labels, perm_mat)

    def plot(self, **kwargs):
        """Plot the distance matrix.

        Parameters
        ----------
        **kwargs
            Arguments passed to ``plot_distmat``.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Plot axes.
        """
        return plot_distmat(self, **kwargs)

    def select(self, labels):
        """Select an induced subset of labels.

        Parameters
        ----------
        labels : sequence
            Labels to retain.

        Returns
        -------
        distances : PairwiseDistances
            Pairwise distances among the selected labels.
        """
        data = [self.get(a, b) for i, a in enumerate(labels) for b in labels[i + 1 :]]

        return self.__class__(labels, np.asarray(data))

    def filter_labels(self, predicate):
        """Select labels satisfying a predicate.

        Parameters
        ----------
        predicate : callable
            Function returning whether a label should be retained.

        Returns
        -------
        distances : PairwiseDistances
            Pairwise distances among retained labels.
        """
        return self.select([label for label in self.labels if predicate(label)])

    def map_labels(self, func):
        """Transform sample labels.

        Parameters
        ----------
        func : callable
            Function applied to each label.

        Returns
        -------
        distances : PairwiseDistances
            Pairwise distances with transformed labels.
        """
        return self.__class__(
            labels=[func(label) for label in self.labels],
            data=self.data,
        )

    @classmethod
    def merge_many(cls, distances):
        """Merge pairwise distance collections.

        Parameters
        ----------
        distances : sequence of PairwiseDistances
            Pairwise distance collections to merge.

        Returns
        -------
        merged : PairDistances
            Distances from all input collections.
        """
        return PairDistances(
            pairs=[pair for dist in distances for pair in dist.pairs],
            data=np.concatenate([dist.data for dist in distances]),
        )
