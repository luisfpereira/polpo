import random
import warnings

from joblib import Parallel, delayed
from tqdm import tqdm

import polpo.utils as putils

from ._preprocessing import (
    Filter,
    FunctionCaller,
    IdentityStep,
    Map,
    StepWrappingPreprocessingStep,
    _wrap_step,
)
from .base import PreprocessingStep


class Hash(PreprocessingStep):
    def __init__(self, key_index=0, ignore_none=False, ignore_empty=False):
        super().__init__()
        self.key_index = key_index
        self.ignore_none = ignore_none
        self.ignore_empty = ignore_empty

    def __call__(self, data):
        new_data = {}
        for datum in data:
            if not isinstance(datum, list):
                datum = list(datum)

            key = datum.pop(self.key_index)

            if len(datum) == 1:
                datum = datum[0]

            if (self.ignore_none and datum is None) or (
                self.ignore_empty and isinstance(datum, list) and len(datum) == 0
            ):
                continue

            new_data[key] = datum

        return new_data


class DictMerger(StepWrappingPreprocessingStep):
    # NB: not shared keys are ignored
    # TODO: allow to raise or provide more behaviors?

    def __init__(self, step=None, as_dict=False):
        # TODO: make as_dict=True the only behavior
        super().__init__(step)
        self.as_dict = as_dict

    def _collect_shared_keys(self, data):
        keys = set(data[0].keys())
        for datum in data[1:]:
            keys = keys.intersection(set(datum.keys()))

        return keys

    def __call__(self, data):
        shared_keys = self._collect_shared_keys(data)
        out = {key: self.step([datum[key] for datum in data]) for key in shared_keys}

        if self.as_dict:
            return out

        return list(out.values())


class HashWithIncoming(StepWrappingPreprocessingStep):
    # TODO: find better name; Hash?
    def __init__(self, step=None, key_step=None, key_subset=None, key_sorter=None):
        if key_subset is not None:
            key_subset = set(key_subset)

        super().__init__(step)
        self.key_step = _wrap_step(key_step)
        self.key_subset = key_subset
        self.key_sorter = key_sorter

    def __call__(self, keys_data):
        values_data = self.step(keys_data)
        keys_data = self.key_step(keys_data)

        zipped_data = zip(keys_data, values_data)

        if self.key_subset is not None:
            zipped_data = (
                (key, value) for key, value in zipped_data if key in self.key_subset
            )

        if self.key_sorter is not None:
            key_sorter = (
                self.key_sorter
                if callable(self.key_sorter)
                else putils.custom_order(self.key_sorter)
            )
            zipped_data = sorted(zipped_data, key=lambda x: key_sorter(x[0]))

        return {key: value for key, value in zipped_data}


class KeySorter(PreprocessingStep):
    def __init__(self, sorter=None):
        if sorter is None:
            sorter = lambda x: x

        self.sorter = sorter
        super().__init__()

    def __call__(self, data):
        sorted_keys = sorted(data.keys(), key=self.sorter)

        return {key: data[key] for key in sorted_keys}


class DictFilter(Filter):
    def __init__(self, func, filter_keys=False):
        if filter_keys:
            func_ = lambda x: func(x[0])
        else:
            func_ = lambda x: func(x[1])
        super().__init__(func_, collection_type=dict, to_iter=lambda x: x.items())


class SelectKeySubset(DictFilter):
    """Select key subset.

    Parameters
    ----------
    subset : array-like
        Subset of keys to consider.
    keep : bool
        Whether to keep or exclude subset.
    """

    def __init__(self, subset, keep=True):
        if keep:
            func = lambda key: key in subset
        else:
            func = lambda key: key not in subset

        super().__init__(func, filter_keys=True)

    def __new__(cls, subset, keep=True):
        """Instantiate step.

        If subset is None, then it instantiates the identity step.
        Syntax sugar to simplify control flow.
        """
        if subset is None:
            return IdentityStep()

        return super().__new__(cls)


class DictNoneRemover(Filter):
    def __init__(self):
        super().__init__(
            func=lambda x: x[1] is not None,
            collection_type=dict,
            to_iter=lambda x: x.items(),
        )


class DictExtractKey(PreprocessingStep):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def __call__(self, data):
        return data[self.key]


class RemoveKeys(PreprocessingStep):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            del data[key]

        return data


class _ExtractUniqueOuterKey(PreprocessingStep):
    def __call__(self, dict_):
        if (n_keys := len(dict_)) != 1:
            raise ValueError(f"Expected one key, but there's {n_keys}")

        return next(iter(dict_.values()))


class ExtractUniqueKey:
    def __new__(cls, nested=False):
        if nested:
            return putils.extract_unique_key_nested

        return _ExtractUniqueOuterKey()


class ExtractRandomValue(PreprocessingStep):
    def __call__(self, data):
        return random.choice(list(data.values()))


class DictToValuesList(PreprocessingStep):
    def __call__(self, data):
        return list(data.values())


class ValuesListToDict(PreprocessingStep):
    def __init__(self, keys=None):
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            keys, values = data
        else:
            keys = self.keys
            values = data

        return dict(zip(keys, values))


class DictToTuplesList(PreprocessingStep):
    def __call__(self, data):
        return list(zip(data.keys(), data.values()))


class DictUpdate(PreprocessingStep):
    def __call__(self, data):
        new_data = data[0].copy()
        for datum in data[1:]:
            new_data.update(datum)

        return new_data


class SerialDictMap(StepWrappingPreprocessingStep):
    """Apply a given step to each element of a dict.

    Parameters
    ----------
    step : callable
        Preprocessing step to apply to value.
    pbar : bool
        Whether to show a progress bar.
    key_step : callable
        Preprocessing step to apply to key.
    special_keys : array-like
        Keys to be subject to a different step.
    special_step : callable
        Step to apply to special keys.
    pass_key : bool
        Whether to pass key to step.
    """

    def __init__(
        self,
        step=None,
        pbar=False,
        key_step=None,
        special_keys=(),
        special_step=None,
        pass_key=False,
    ):
        super().__init__(step)
        self.pbar = pbar
        self.key_step = _wrap_step(key_step)
        self.special_keys = special_keys
        self.special_step = _wrap_step(special_step)
        self.pass_key = pass_key

    def __call__(self, data):
        """Apply step.

        Parameters
        ----------
        data : dict

        Returns
        -------
        new_data : dict
        """
        out = {}
        kwargs = {}
        for key, value in tqdm(data.items(), disable=not self.pbar):
            if self.pass_key:
                kwargs = {"key": key}

            if key in self.special_keys:
                value_ = self.special_step(value, **kwargs)
            else:
                value_ = self.step(value, **kwargs)

            out[self.key_step(key)] = value_

        return out


class ParDictMap(StepWrappingPreprocessingStep):
    """Apply a given step to each element of a dict in parallel.

    Parameters
    ----------
    step : callable
        Preprocessing step to apply to value.
    n_jobs : int
        The maximum number of concurrently running jobs.
    verbose : int
        The verbosity level: if non zero, progress messages are
        printed. Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.

    Notes
    -----
    * has less functionality than `DictMap`
    """

    def __init__(self, step, n_jobs=-1, verbose=0):
        super().__init__(step)
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __call__(self, data):
        with Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            res = parallel(delayed(self.step)(datum) for datum in data.items())

        return dict(zip(data.keys(), res))


class DictMap:
    """Apply a given step to each element of a dict."""

    def __new__(
        cls,
        step=None,
        pbar=False,
        key_step=None,
        special_keys=(),
        special_step=None,
        pass_key=False,
        n_jobs=0,
        verbose=0,
    ):
        """Instantiate class.

        Parameters
        ----------
        step : callable
            Preprocessing step to apply to value.
        pbar : bool
            Whether to show a progress bar. Only if serial.
        key_step : callable
            Preprocessing step to apply to key. Only if serial.
        special_keys : array-like
            Keys to be subject to a different step. Only if serial.
        special_step : callable
            Step to apply to special keys. Only if serial.
        pass_key : bool
            Whether to pass key to step. Only if serial.
        n_jobs : int
            The maximum number of concurrently running jobs.
        verbose : int
            The verbosity level. Only if parallel.
        """
        if n_jobs != 0:
            if pbar or key_step is not None or special_keys or special_step or pass_key:
                warnings.warn(
                    "Several arguments where ignored, check `ParDictMap` for more info"
                )

            return ParDictMap(step, n_jobs=n_jobs, verbose=verbose)

        return SerialDictMap(
            step=step,
            pbar=pbar,
            key_step=key_step,
            special_keys=special_keys,
            special_step=special_step,
            pass_key=pass_key,
        )


class NestedDictMap:
    """Nested dict map.

    Parameters
    ----------
    depth : int
        Depth at which step is applied.
    inner_is_dict : bool
        Whether inner is a dict. If not, must be an iterable.
    """

    def __new__(self, step, *args, depth=1, inner_is_dict=True, **kwargs):
        step = (
            DictMap(step, *args, **kwargs)
            if inner_is_dict
            else Map(step, *args, **kwargs)
        )
        for _ in range(depth):
            step = DictMap(step)

        return step


class TransformValues(StepWrappingPreprocessingStep):
    # \left(V_1, \ldots, V_n\right) \mapsto\left(W_1, \ldots, W_n\right)
    def __call__(self, data):
        keys = list(data.keys())
        out = self.step(data.values())
        return dict(zip(keys, out))


class RenameKeys(SerialDictMap):
    def __init__(self, key_map):
        if isinstance(key_map, dict):
            key_map_ = key_map
            key_map = lambda key: key_map_.get(key, key)

        super().__init__(key_step=key_map)


class RenameNestedKeys(PreprocessingStep):
    def __init__(self, key_map):
        super().__init__()
        self.key_map = key_map

    def __call__(self, data):
        new_data = {}
        for key, dict_ in data.items():
            new_data[key] = RenameKeys(self.key_map.get(key))(dict_)

        return new_data


class NestedDictToList(PreprocessingStep):
    def __init__(self, key_ids):
        super().__init__()
        self.key_ids = key_ids

    def __call__(self, data):
        ls_elem = {}
        out_ls = []

        def _rec_func(datum, key_ids):
            if len(key_ids) == 1:
                ls_elem[key_ids[0]] = datum
                out_ls.append(ls_elem.copy())

                return

            for key, values in datum.items():
                ls_elem[key_ids[0]] = key
                _rec_func(values, key_ids[1:])

            return

        _rec_func(data, self.key_ids)

        return out_ls


class NestedDictSwapper(PreprocessingStep):
    def __call__(self, nested_dict):
        return {
            outer_key: {
                inner_key: nested_dict[inner_key][outer_key]
                for inner_key in nested_dict
            }
            for outer_key in next(iter(nested_dict.values()))
        }


class ListDictSwapper(PreprocessingStep):
    # swap a list of dict
    # assumes same keys

    def __call__(self, ls):
        if len(ls) == 0:
            return {}

        keys = ls[0].keys()
        out = {key: [] for key in keys}
        for elem in ls:
            for key in keys:
                out[key].append(elem[key])

        return out


class DictListSwapper(PreprocessingStep):
    # swap a dict of list
    # assumes lists have the same size

    def __call__(self, data):
        out = [{} for _ in range(len(data[list(data.keys())[0]]))]
        n_out = len(out)

        for key, value_ls in data.items():
            if len(value_ls) != n_out:
                raise ValueError("Trying to swap lists with different sizes.")

            for index, value in enumerate(value_ls):
                out[index][key] = value

        return out


class ZipWithKeys(StepWrappingPreprocessingStep):
    """Apply step to the values of a dict.

    Then, get a list back, zip with original keys, and return a dict.
    """

    def __init__(self, step, check_length=True):
        super().__init__(step)
        self.check_length = check_length

    def __call__(self, dict_):
        keys = list(dict_.keys())
        values = list(dict_.values())
        new_values = self.step(values)

        if self.check_length and len(new_values) != len(keys):
            raise ValueError(
                f"Length mismatch: {len(keys)} keys but {len(new_values)} outputs"
            )

        return dict(zip(keys, new_values))


class UnnestDict(FunctionCaller):
    def __init__(self, sep="/"):
        super().__init__(putils.unnest_dict, sep=sep)


class NestDict(FunctionCaller):
    def __init__(self, sep="/"):
        super().__init__(putils.nest_dict, sep=sep)


class TruncateDict(PreprocessingStep):
    def __init__(self, n_keys=None):
        super().__init__()
        self.n_keys = n_keys

    def __call__(self, data):
        if self.n_keys is None:
            return data

        keys = list(data.keys())
        keys_to_keep = keys[: min(len(keys), self.n_keys)]

        return {key: data[key] for key in keys_to_keep}


class Subsample(PreprocessingStep):
    def __init__(self, n_samples=None):
        super().__init__()
        self.n_samples = n_samples

    def __call__(self, data):
        if self.n_samples is None:
            return data

        keys = random.sample(list(data.keys()), k=self.n_samples)
        return {key: data[key] for key in keys}
