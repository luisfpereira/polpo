import numpy as np

from polpo.dataset import Dataset


class BasePairDistances:
    """Collection of scalar distances indexed by label pairs."""

    def get(self, label_a, label_b):
        """Return the distance between two labels."""
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
