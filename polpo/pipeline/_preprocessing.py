import abc
import itertools
import logging
import warnings
from functools import wraps

from joblib import Parallel, delayed
from tqdm import tqdm

from polpo.utils import is_non_string_iterable, unnest

from .base import IdentityStep, Pipeline, PreprocessingStep


def _wrap_step(step=None):
    if step is None:
        step = IdentityStep()

    if is_non_string_iterable(step):
        step = Pipeline(step)

    return step


def pipe_to_func(pipe_cls):
    @wraps(pipe_cls)
    def func(*args, **kwargs):
        return pipe_cls(*args, **kwargs)()

    return func


class StepWrappingPreprocessingStep(PreprocessingStep, abc.ABC):
    def __init__(self, step=None):
        super().__init__()

        self.step = _wrap_step(step)


class ExceptionToWarning(StepWrappingPreprocessingStep):
    def __init__(self, step, warn=True):
        super().__init__(step)
        self.warn = warn

    def __call__(self, data):
        try:
            return self.step(data)
        except Exception as e:
            if self.warn:
                warnings.warn(str(e))

        return data


class BranchingPipeline(PreprocessingStep):
    def __init__(self, branches, merger=None):
        if merger is None:
            # TODO: default to identity?
            merger = NestingSwapper()

        super().__init__()
        self.branches = branches
        self.merger = _wrap_step(merger)

    def __call__(self, data=None):
        out = []
        for pipeline in self.branches:
            out.append(pipeline(data))

        return self.merger(out)


class NestingSwapper(PreprocessingStep):
    def __call__(self, data):
        return list(zip(*data))


class IndexSelector(StepWrappingPreprocessingStep):
    def __init__(self, index=0, repeat=False, step=None):
        super().__init__(step)

        self.index = index
        self.repeat = repeat

    def __call__(self, data):
        selected = self.step(data[self.index])
        if self.repeat:
            return [selected] * len(data)

        return selected


class RemoveIndex(PreprocessingStep):
    def __init__(self, index=0, inplace=False):
        super().__init__()
        self.index = index
        self.inplace = inplace

    def __call__(self, data):
        if not self.inplace:
            data = data.copy()

        data.pop(self.index)
        return data


class Unnest(PreprocessingStep):
    def __call__(self, data):
        return unnest(data)


class ListSqueeze(PreprocessingStep):
    def __init__(self, raise_=True):
        self.raise_ = raise_

    def __call__(self, data):
        if len(data) != 1:
            if self.raise_:
                raise ValueError("Unsqueezable!")
            else:
                return data

        return data[0]


class TupleWith(StepWrappingPreprocessingStep):
    def __init__(self, step, incoming_first=True):
        super().__init__(step)
        self.incoming_first = incoming_first

    def __call__(self, data):
        new_data = self.step(data)
        ordering = (data, new_data) if self.incoming_first else (new_data, data)
        return [(datum, datum_) for datum, datum_ in zip(*ordering)]


class TupleWithIncoming(TupleWith):
    def __init__(self, step):
        super().__init__(step, incoming_first=True)


class Filter(PreprocessingStep):
    def __init__(self, func, collection_type=list, to_iter=None):
        if to_iter is None:
            to_iter = lambda x: x

        super().__init__()
        self.func = func
        self.collection_type = collection_type
        self.to_iter = to_iter

    def __call__(self, data):
        return self.collection_type(filter(self.func, self.to_iter(data)))


class EmptyRemover(Filter):
    def __init__(self):
        super().__init__(func=lambda x: len(x) > 0)


class NoneRemover(Filter):
    def __init__(self):
        super().__init__(func=lambda x: x is not None)


class NoneSkipper(StepWrappingPreprocessingStep):
    def __call__(self, data):
        if data is None:
            return data

        return self.step(data)


class EmptySkipper(StepWrappingPreprocessingStep):
    def __call__(self, data):
        if len(data) == 0:
            return data

        return self.step(data)


class SerialMap(StepWrappingPreprocessingStep):
    def __init__(self, step, pbar=False):
        super().__init__(step)
        self.pbar = pbar

    def __call__(self, data):
        return [self.step(datum) for datum in tqdm(data, disable=not self.pbar)]


class ParMap(StepWrappingPreprocessingStep):
    def __init__(self, step, n_jobs=-1, verbose=0):
        super().__init__(step)
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __call__(self, data):
        with Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            res = parallel(delayed(self.step)(datum) for datum in data)

        return list(res)


class EnsureIterable(StepWrappingPreprocessingStep):
    def _is_iter(self, data):
        return is_non_string_iterable(data)

    def __call__(self, data):
        decorated = False
        if not self._is_iter(data):
            decorated = True
            data = [data]

        new_data = self.step(data)

        if decorated:
            if isinstance(new_data, dict):
                return new_data[list(new_data.keys())[0]]

            return new_data[0]

        return new_data


class EnsureListIterable(EnsureIterable):
    def _is_iter(self, data):
        return isinstance(data, list)


class Map:
    def __new__(cls, step, n_jobs=1, verbose=0, force_iter=False):
        if step is None or isinstance(step, IdentityStep):
            return IdentityStep()

        if n_jobs != 1:
            map_ = ParMap(step, n_jobs=n_jobs, verbose=verbose)

        else:
            map_ = SerialMap(step, pbar=verbose > 0)

        if force_iter:
            return EnsureIterable(map_)

        return map_


class IndexMap(StepWrappingPreprocessingStep):
    # TODO: rename to `MapAtIndex`?
    # TODO: name is confusing
    def __init__(self, step, index=0):
        super().__init__(step)
        self.index = index

    def __call__(self, data):
        if isinstance(data, tuple):
            data = list(data)

        data[self.index] = self.step(data[self.index])

        return data


class DataPrinter(PreprocessingStep):
    # useful for debugging

    def __init__(self, silent=False, headers=None):
        super().__init__()
        # avoids having to comment code out
        self.silent = silent
        self.headers = headers

    def __call__(self, data):
        if not self.silent:
            if self.headers is not None:
                print(self.headers)

            print(data)
        return data


class PartiallyInitializedStep(PreprocessingStep):
    """Instantiate a step based on data.

    Parameters
    ----------
    Step : PreprocessingStep
        Step to be instantiated.
    pass_data : bool
        Whether to pass that to callable after instantiation.
    """

    def __init__(self, Step, pass_data=True, **kwargs):
        super().__init__()
        self.Step = Step
        self.pass_data = pass_data
        self.kwargs = kwargs

    def instantiate(self, data):
        kwargs = self.kwargs.copy()
        dependent_keys = list(filter(lambda x: x.startswith("_"), kwargs.keys()))
        for key in dependent_keys:
            value = kwargs.pop(key)
            if callable(value):
                value = value(data)

            kwargs[key[1:]] = value

        return self.Step(**kwargs)

    def __call__(self, data):
        step = self.instantiate(data)

        if self.pass_data:
            return step(data)

        return step()


class IfCondition(StepWrappingPreprocessingStep):
    def __init__(self, step, else_step, condition):
        super().__init__(step)
        self.else_step = _wrap_step(else_step)
        self.condition = condition

    def __call__(self, data):
        if self.condition(data):
            return self.step(data)

        return self.else_step(data)


class IfEmpty(IfCondition):
    def __init__(self, step, else_step):
        super().__init__(step, else_step, condition=lambda x: len(x) == 0)


class Constant(PreprocessingStep):
    """Constant.

    Parameters
    ----------
    value : any
        Constant.
    """

    def __init__(self, value):
        super().__init__()
        self.value = value

    def __call__(self, value=None):
        """Apply step.

        Parameters
        ----------
        value : str
            Value.

        Returns
        -------
        constant : any
        """
        return value if value is not None else self.value


class MethodApplier(PreprocessingStep):
    """Applies a named method with preset args and kwargs.

    Parameters
    ----------
    method : str
        Named method.
    """

    def __init__(self, *args, method, **kwargs):
        super().__init__()
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def __call__(self, obj):
        """Apply step.

        Parameters
        ----------
        obj : object

        Returns
        -------
        bool
        """
        return getattr(obj, self.method)(*self.args, **self.kwargs)


class FunctionCaller(PreprocessingStep):
    """Calls a function with preset kwargs.

    Parameters
    ----------
    func : callable
        Function.
    """

    def __init__(self, func, **kwargs):
        self.func = lambda x: func(x, **kwargs)

    def __call__(self, data=None):
        return self.func(data)


class StepWithLogging(StepWrappingPreprocessingStep):
    # TODO: control logging level
    def __init__(self, step, msg):
        super().__init__(step)
        self.msg = msg

    def __call__(self, data=None):
        logging.info(self.msg)
        return self.step(data)


class GroupBy(PreprocessingStep):
    """Group data.

    Parameters
    ----------
    datum2group : callable
    """

    def __init__(self, datum2group):
        super().__init__()
        self.datum2group = datum2group

    def __call__(self, data):
        out = {}
        for datum in data:
            key = self.datum2group(datum)
            key_ls = out.get(key, [])
            key_ls.append(datum)
            out[key] = key_ls

        return out


class FilteredGroupBy(PreprocessingStep):
    """Group data.

    Parameters
    ----------
    datum2group : callable
    subset : set
        Keys not belonging to ``subset`` are filtered out.
    """

    def __init__(self, datum2group, subset):
        super().__init__()
        self.datum2group = datum2group
        self.subset = subset

    def __new__(cls, datum2group, subset=None):
        if subset is None:
            return GroupBy(datum2group)

        return super().__new__(cls)

    def __call__(self, data):
        out = {key: [] for key in self.subset}
        for datum in data:
            key = self.datum2group(datum)

            if key not in out:
                continue

            out[key].append(datum)

        return out


class InjectData(PreprocessingStep):
    def __init__(self, data, as_first=True):
        super().__init__()
        self.data = data
        self.as_first = as_first

    def __call__(self, data):
        if self.as_first:
            return (self.data, data)

        return (data, self.data)


class CartesianProduct(PreprocessingStep):
    def __call__(self, data):
        return list(itertools.product(data[0], data[1]))
