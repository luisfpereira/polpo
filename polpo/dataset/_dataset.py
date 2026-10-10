from collections.abc import Mapping

import numpy as np

from polpo.utils import nest_dict, unnest_dict


class DatasetMapping(Mapping):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"{type(self).__name__}({self.data!r})"

    def _new(self, data):
        return type(self)(data)

    def keys_list(self):
        return list(self.data.keys())

    def as_dict(self):
        return self.data

    def merge(self, other):
        return type(self).merge_many([self, other])

    def sort_keys(self, key=None, reverse=False):
        """Sort dataset items by key."""
        sort_key = lambda item: item[0] if key is None else key(item[0])
        data = dict(sorted(self.data.items(), key=sort_key, reverse=reverse))

        return type(self)(data)


class Dataset(DatasetMapping):
    def values_list(self):
        return list(self.data.values())

    def with_values(self, values):
        # uses same keys
        data = dict(zip(self.data.keys(), values))
        return Dataset(data)

    def nest(self):
        data = nest_dict(self.data)
        return NestedDataset(data)

    def map_values(self, func, /, *args, **kwargs):
        """Apply ``func`` independently to every dataset value.

        Parameters
        ----------
        func : callable
            Function applied to each value.
        *args
            Additional positional arguments passed to ``func``.
        **kwargs
            Additional keyword arguments passed to ``func``.

        Returns
        -------
        Dataset
            Dataset with the same keys and transformed values.
        """
        data = {key: func(value, *args, **kwargs) for key, value in self.data.items()}
        return self._new(data)

    def map_items(self, func, /, *args, **kwargs):
        """Apply ``func`` independently to every dataset item.

        Parameters
        ----------
        func : callable
            Function applied to each key-value pair.
        *args
            Additional positional arguments passed to ``func``.
        **kwargs
            Additional keyword arguments passed to ``func``.

        Returns
        -------
        Dataset
            Dataset with the same keys and transformed values.
        """
        data = {
            key: func(key, value, *args, **kwargs) for key, value in self.data.items()
        }
        return self._new(data)

    def map_keys(self, func, /, *args, on_collision="raise", **kwargs):
        """Apply ``func`` to each key while preserving values."""
        data = {}

        for key, value in self.items():
            new_key = func(key, *args, **kwargs)

            if new_key in data:
                if on_collision == "raise":
                    raise ValueError(
                        f"Key transformation produced duplicate key {new_key!r}."
                    )
                if on_collision == "keep_first":
                    continue
                if on_collision != "keep_last":
                    raise ValueError(
                        "on_collision must be 'raise', 'keep_first', or 'keep_last'."
                    )

            data[new_key] = value

        return self._new(data)

    def apply(self, func, /, *args, **kwargs):
        """Apply ``func`` once to the ordered dataset values."""
        return func(self.values_list(), *args, **kwargs)

    def transform(self, func, /, *args, **kwargs):
        """Apply ``func`` to all values and preserve the dataset keys."""
        return self.with_values(self.apply(func, *args, **kwargs))

    def sample(self, n_samples=1, *, random_state=None):
        """Sample dataset entries without replacement.

        Parameters
        ----------
        n_samples : int
            Number of entries to sample.
        random_state : int or numpy.random.Generator, optional
            Random seed or generator.

        Returns
        -------
        Dataset
            Dataset containing the sampled entries.
        """
        if n_samples > len(self):
            raise ValueError(
                f"Cannot sample {n_samples} entries from a dataset "
                f"containing {len(self)}."
            )

        rng = np.random.default_rng(random_state)

        keys = self.keys_list()
        indices = rng.choice(
            len(keys),
            size=n_samples,
            replace=False,
        )
        sampled_keys = [keys[index] for index in indices]

        return self._new({key: self[key] for key in sampled_keys})

    def select(self, keys, *, ignore_missing=False):
        """Return a dataset restricted to the given keys.

        Parameters
        ----------
        keys : iterable
            Keys to select. Their order determines the order of the returned
            dataset.

        Returns
        -------
        Dataset
            Dataset containing the selected entries.
        """
        if ignore_missing:
            return self._new({key: self[key] for key in keys if key in self})

        return self._new({key: self[key] for key in keys})

    def drop(self, keys):
        """Drop entries by key.

        Parameters
        ----------
        keys : collection
            Keys to drop.

        Returns
        -------
        dataset : Dataset
            Dataset without the specified entries.
        """
        keys = set(keys)
        return self.select([key for key in self.keys() if key not in keys])

    def filter_values(self, predicate):
        return self._new(
            dict((key, value) for key, value in self.items() if predicate(value))
        )

    @classmethod
    def zip_many(cls, datasets, func):
        if not datasets:
            return cls({})

        keys = datasets[0].keys()

        if any(dataset.keys() != keys for dataset in datasets[1:]):
            raise ValueError("Datasets do not have matching keys.")

        return cls({key: func([dataset[key] for dataset in datasets]) for key in keys})

    @classmethod
    def merge_many(cls, datasets):
        data = {}

        for dataset in datasets:
            overlap = data.keys() & dataset.keys()
            if overlap:
                raise ValueError(f"Duplicate keys: {overlap}")

            data.update(dataset.items())

        return cls(data)

    @classmethod
    def from_keys(cls, keys, func):
        """Create a dataset by evaluating a function at each key."""
        return cls({key: func(key) for key in keys})


class NestedDataset(DatasetMapping):
    def nested_keys(self):
        return {outer_key: list(inner) for outer_key, inner in self.items()}

    def key_pairs(self):
        return [
            (outer_key, inner_key)
            for outer_key, inner in self.items()
            for inner_key in inner
        ]

    def flatten(self):
        data = unnest_dict(self.data, sep=None)
        return Dataset(data)

    def transform(self, func, /, *args, **kwargs):
        """Apply a function to the flattened dataset and restore its structure.

        The dataset is flattened into an ordered list of values and passed as the
        first argument to ``func``.
        The values returned by ``func`` are associated
        with the original keys and converted back into a nested dataset.

        Parameters
        ----------
        func : callable
            Function applied to the flattened values. Its first argument must
            accept the list of dataset values.
        *args
            Additional positional arguments forwarded to ``func``.
        **kwargs
            Additional keyword arguments forwarded to ``func``.

        Returns
        -------
        NestedDataset
            A nested dataset containing the values returned by ``func``.
        """
        return self.flatten().transform(func, *args, **kwargs).nest()

    def map_values(self, func, /, *args, **kwargs):
        """Apply ``func`` independently to every inner value.

        Parameters
        ----------
        func : callable
            Function applied to each inner value.
        *args
            Additional positional arguments passed to ``func``.
        **kwargs
            Additional keyword arguments passed to ``func``.

        Returns
        -------
        NestedDataset
            Dataset with the same keys and transformed values.
        """
        data = {
            outer_key: {
                inner_key: func(value, *args, **kwargs)
                for inner_key, value in inner_data.items()
            }
            for outer_key, inner_data in self.data.items()
        }
        return self._new(data)

    def map_items(self, func, /, *args, **kwargs):
        """Apply ``func`` to each outer key, inner key, and value."""
        data = {
            outer_key: {
                inner_key: func(
                    outer_key,
                    inner_key,
                    value,
                    *args,
                    **kwargs,
                )
                for inner_key, value in inner_data.items()
            }
            for outer_key, inner_data in self.items()
        }
        return self._new(data)

    def map_keys(self, func, /, *args, **kwargs):
        data = {}

        for outer_key, inner_data in self.items():
            for inner_key, value in inner_data.items():
                new_outer_key, new_inner_key = func(
                    outer_key,
                    inner_key,
                    *args,
                    **kwargs,
                )
                data.setdefault(new_outer_key, {})[new_inner_key] = value

        return self._new(data)

    def reduce_outer(self, func, /, *args, **kwargs):
        """Apply ``func`` to each outer dataset and return one result per key."""
        return Dataset(
            {
                outer_key: func(list(inner_data.values()), *args, **kwargs)
                for outer_key, inner_data in self.data.items()
            }
        )

    def sample_inner(self, n_samples=1, *, random_state=None):
        """Sample inner entries independently for each outer key.

        Parameters
        ----------
        n_samples : int
            Number of inner entries sampled per outer key.
        random_state : int or numpy.random.Generator
            Random seed or generator.

        Returns
        -------
        NestedDataset
            Dataset containing the sampled inner entries.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")

        rng = np.random.default_rng(random_state)

        data = {}

        for outer_key, inner_data in self.items():
            inner_keys = list(inner_data)

            if n_samples > len(inner_keys):
                raise ValueError(
                    f"Cannot sample {n_samples} entries from "
                    f"{outer_key!r}, which contains {len(inner_keys)}."
                )

            indices = rng.choice(
                len(inner_keys),
                size=n_samples,
                replace=False,
            )
            sampled_keys = [inner_keys[index] for index in indices]

            data[outer_key] = {
                inner_key: inner_data[inner_key] for inner_key in sampled_keys
            }

        return self._new(data)

    def filter_keys(self, predicate, /, *args, **kwargs):
        """Filter entries according to their outer and inner keys.

        Parameters
        ----------
        predicate : callable
            Function called as ``predicate(outer_key, inner_key, *args, **kwargs)``.
            Entries for which it returns ``True`` are retained.
        *args
            Additional positional arguments passed to ``predicate``.
        **kwargs
            Additional keyword arguments passed to ``predicate``.

        Returns
        -------
        NestedDataset
            Dataset containing the selected entries.
        """
        data = {
            outer_key: {
                inner_key: value
                for inner_key, value in inner_data.items()
                if predicate(outer_key, inner_key, *args, **kwargs)
            }
            for outer_key, inner_data in self.items()
        }

        return self._new(
            {
                outer_key: inner_data
                for outer_key, inner_data in data.items()
                if inner_data
            }
        )

    def drop_outer(self, keys):
        keys = set(keys)
        return self._new(
            {
                outer_key: inner
                for outer_key, inner in self.items()
                if outer_key not in keys
            }
        )

    def split_outer(self):
        """Split into one Dataset per outer key."""
        return Dataset({outer_key: self.get_outer(outer_key) for outer_key in self})

    def iter_outer(self):
        for outer_key in self:
            yield outer_key, self.get_outer(outer_key)

    def get_outer(self, outer_key):
        """Return the inner dataset associated with an outer key."""
        return Dataset(self.data[outer_key])

    def select_outer(self, keys):
        keys = set(keys)

        return type(self)(
            {
                outer_key: inner_data
                for outer_key, inner_data in self.data.items()
                if outer_key in keys
            }
        )

    def group_outer(self, grouper):
        groups = {}

        for outer_key, outer_data in self.iter_outer():
            group = grouper(outer_key)
            groups.setdefault(group, {})[outer_key] = outer_data

        return Dataset({group: type(self)(data) for group, data in groups.items()})

    def select_inner(self, keys):
        """Select inner entries for each outer key."""
        return NestedDataset(
            {
                outer_key: Dataset(values).select(keys[outer_key]).data
                for outer_key, values in self.items()
            }
        )

    def drop_inner(self, keys):
        """Drop inner entries for each outer key."""
        return NestedDataset(
            {
                outer_key: Dataset(values).drop(keys.get(outer_key, [])).data
                for outer_key, values in self.items()
            }
        )

    def sort_inner_keys(self, key=None, reverse=False):
        return type(self)(
            {
                outer_key: dict(
                    sorted(
                        inner_data.items(),
                        key=lambda item: key(item[0]) if key is not None else item[0],
                        reverse=reverse,
                    )
                )
                for outer_key, inner_data in self.items()
            }
        )

    def to_dataframe(
        dataset,
        outer_col="subject",
        inner_col="time",
        value_col="Y",
        **constants,
    ):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    outer_col: outer_key,
                    inner_col: inner_key,
                    value_col: value,
                    **constants,
                }
                for outer_key in dataset.keys()
                for inner_key, value in dataset.get_outer(outer_key).items()
            ]
        )

    @classmethod
    def from_dataframe(cls, data, outer_col, inner_col, value_col):
        if data.duplicated([outer_col, inner_col]).any():
            raise ValueError(
                f"Columns {outer_col!r} and {inner_col!r} do not uniquely identify rows."
            )

        return cls(
            {
                outer_value: dict(zip(group[inner_col], group[value_col]))
                for outer_value, group in data.groupby(outer_col, sort=False)
            }
        )

    @classmethod
    def zip_many(cls, datasets, func):
        return Dataset.zip_many(
            [dataset.flatten() for dataset in datasets], func
        ).nest()

    @classmethod
    def merge_many(cls, datasets):
        data = {}

        for dataset in datasets:
            for outer_key, inner_dataset in dataset.items():
                if outer_key in data:
                    inner_dataset = Dataset(data[outer_key]).merge(inner_dataset).data

                data[outer_key] = inner_dataset

        return cls(data)

    @classmethod
    def from_keys(cls, nested_keys, func):
        """Create a nested dataset by evaluating a function at each key."""
        return cls(
            {
                outer_key: {
                    inner_key: func(outer_key, inner_key) for inner_key in inner_keys
                }
                for outer_key, inner_keys in nested_keys.items()
            }
        )
