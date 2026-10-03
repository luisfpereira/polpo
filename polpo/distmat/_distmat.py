import numpy as np

from polpo.dataset import Dataset
from polpo.ext.numpy.io import load_indexed_array, save_indexed_array
from polpo.utils.np import triu_vec_to_sym

from .plot import plot_dist_mat


def permute_by_row_norm(mat, descending=True):
    """Permute both matrix axes according to row norm.

    Parameters
    ----------
    mat : array-like, shape=(n, n)
        Square matrix.
    descending : bool
        Whether to order by decreasing row norm.

    Returns
    -------
    permuted_mat : array-like, shape=(n, n)
        Matrix with rows and columns permuted.
    indices : array-like, shape=(n,)
        Indices defining the permutation.
    """
    # checks for global isolation
    row_norms = np.linalg.norm(mat, axis=-1)

    signal = -1.0 if descending else 1.0
    sorted_idx = np.argsort(signal * row_norms, axis=-1)

    perm_mat = mat[np.ix_(sorted_idx, sorted_idx)]

    return perm_mat, sorted_idx


def knn_scores(mat, k=5):
    """Compute mean distances to nearest neighbors.

    Parameters
    ----------
    mat : array-like, shape=(n, n)
        Pairwise distance matrix.
    k : int
        Number of nearest neighbors.

    Returns
    -------
    scores : array-like, shape=(n,)
        Mean nearest-neighbor distance for each sample.
    """
    # mat: (n, n) distance matrix
    idx = np.argsort(mat, axis=1)
    knn = np.take_along_axis(mat, idx[:, 1 : k + 1], axis=1)
    return knn.mean(axis=1)


class BasePairDistances:
    """Collection of scalar distances indexed by label pairs."""

    def get(self, label_a, label_b):
        raise NotImplementedError

    def items(self):
        """Iterate over pairs and distances.

        Returns
        -------
        items : iterator
            Pairs and corresponding distances.
        """
        return zip(self.pairs, self.data)

    def as_dataset(self):
        """Return distances as a pair-keyed dataset.

        Returns
        -------
        dataset : Dataset
            Dataset mapping label pairs to distances.
        """
        return Dataset(dict(self.items()))

    def select_pairs(self, pairs):
        """Select distances for arbitrary pairs.

        Parameters
        ----------
        pairs : sequence, shape=(n_pairs, 2)
            Pairs to select.

        Returns
        -------
        distances : PairDistances
            Distances associated with the selected pairs.
        """
        return PairDistances(
            pairs=pairs,
            data=np.asarray([self.get(*pair) for pair in pairs]),
        )

    def group_pairs(self, grouper):
        """Group pairwise distances according to their label pairs.

        Parameters
        ----------
        grouper : callable
            Function mapping two labels to a group key.

        Returns
        -------
        groups : dict
            Mapping group keys to ``PairDistances``.
        """
        groups = {}

        for pair in self.pairs:
            group = grouper(*pair)
            groups.setdefault(group, []).append(pair)

        return {group: self.select_pairs(pairs) for group, pairs in groups.items()}


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
        indices = np.triu_indices(len(labels), k=1)
        return cls(labels, matrix[indices])

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
        """Return the condensed index for a pair.

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
        index : int or None
            Condensed index, or ``None`` for identical indices.
        """
        if i == j:
            return None

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
        perm_mat, sorted_idx = permute_by_row_norm(self.matrix)
        labels = [self.labels[index] for index in sorted_idx]
        return self.__class__.from_matrix(labels, perm_mat)

    def plot(self, **kwargs):
        """Plot the distance matrix.

        Parameters
        ----------
        **kwargs
            Arguments passed to ``plot_dist_mat``.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Plot axes.
        """
        return plot_dist_mat(self, **kwargs)

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


class PairDistances(BasePairDistances):
    """Distances associated with arbitrary pairs.

    Parameters
    ----------
    pairs : sequence, shape=(n_pairs, 2)
        Pairs associated with distances.
    data : array-like, shape=(n_pairs,)
        Distance values.
    """

    def __init__(self, pairs, data):
        if len(pairs) != len(data):
            raise ValueError("pairs and data must have the same length.")

        self.pairs = pairs
        self.data = data

        self._pair_to_index = {}
        for index, (a, b) in enumerate(self.pairs):
            self._pair_to_index[a, b] = index
            self._pair_to_index[b, a] = index

    @property
    def labels(self):
        """Return labels appearing in pairs.

        Returns
        -------
        labels : list
            Unique labels in order of appearance.
        """
        return list(dict.fromkeys(label for pair in self.pairs for label in pair))

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
        return self.data[self._pair_to_index[label_a, label_b]]

    def select(self, labels):
        """Select distances involving only the given labels.

        Parameters
        ----------
        labels : collection
            Labels to retain.

        Returns
        -------
        distances : PairDistances
            Distances whose pair labels are both selected.
        """
        labels = set(labels)
        mask = np.asarray([a in labels and b in labels for a, b in self.pairs])

        return self.__class__(
            pairs=[pair for pair, keep in zip(self.pairs, mask) if keep],
            data=self.data[mask],
        )
