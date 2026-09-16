import functools
import random
from collections.abc import Mapping
from dataclasses import dataclass

from geomstats.test.random import get_data_generator

from .execution import get_execution_key


@dataclass(frozen=True)
class TestDatum:
    data: object
    marks: tuple = ()


def _unpack_test_datum(datum):
    if isinstance(datum, TestDatum):
        return datum.data, datum.marks

    return datum, ()


def _normalize_datum(datum, arg_names):
    if isinstance(datum, Mapping):
        unknown = datum.keys() - set(arg_names)
        if unknown:
            raise ValueError(f"Unknown argument names: {sorted(unknown)}.")

        return datum

    if len(datum) > len(arg_names):
        raise ValueError(f"Got {len(datum)} values for {len(arg_names)} arguments.")

    return dict(zip(arg_names, datum))


class TestData:
    suffix = "_test_data"

    def get_data_methods(self):
        """Return available test-data providers indexed by vanilla test name."""
        data = {}

        for name in dir(self):
            if not name.endswith(self.suffix):
                continue

            data[name.removesuffix(self.suffix)] = getattr(self, name)

        return data

    def get_decorators(self):
        return ()


class _BaseLazyValue:
    """Base class for lazily evaluated test values.

    Parameters
    ----------
    func : callable
        Function used to compute the value.
    *args
        Positional arguments passed to ``func`` when the value is evaluated.
        Lazy arguments are materialized recursively.
    **kwargs
        Keyword arguments passed to ``func`` when the value is evaluated.
        Lazy arguments are materialized recursively.
    """

    def __init__(self, func, *args, label=None, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.label = label

    def __repr__(self):
        if self.label is not None:
            return self.label

        return super().__repr__()

    def _evaluate(self):
        args = [_materialize(arg) for arg in self.args]
        kwargs = {name: _materialize(value) for name, value in self.kwargs.items()}
        return self.func(*args, **kwargs)

    def materialize(self):
        raise NotImplementedError


class LazyValue(_BaseLazyValue):
    """Lazily evaluated value cached for each test execution.

    The wrapped function is evaluated at most once for each execution key.
    Repeated materialization within the same execution returns the cached
    value, while a new execution causes the value to be evaluated again.
    """

    def __init__(self, func, *args, **kwargs):
        super().__init__(func, *args, **kwargs)
        self._values = {}

    def materialize(self):
        key = get_execution_key()

        if key not in self._values:
            self._values[key] = self._evaluate()

        return self._values[key]


class DynamicLazyValue(_BaseLazyValue):
    """Lazily evaluated value recomputed on every materialization.

    Unlike :class:`LazyValue`, the wrapped function is evaluated every time
    ``materialize`` is called.
    """

    def materialize(self):
        return self._evaluate()


def _materialize(value):
    if isinstance(value, _BaseLazyValue):
        return value.materialize()

    return value


def materialize_lazy_values(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = tuple(_materialize(value) for value in args)
        kwargs = {name: _materialize(value) for name, value in kwargs.items()}
        return func(*args, **kwargs)

    return wrapper


class ManifoldTestData(TestData):
    # TODO: move to geomstats
    RANDOM_POINT_COUNTS = [1] + random.sample(range(2, 5), 1)

    def __init__(self, space=None):
        self._space = None
        self.data_generator = None

        if space is not None:
            self.space = space

    @property
    def space(self):
        return self._space

    @space.setter
    def space(self, space):
        self._space = space
        self.data_generator = get_data_generator(space)

    def get_decorators(self):
        return [materialize_lazy_values]

    def generate_random_data(
        self,
        arg_names,
        dependencies=None,
        exclude_single=False,
        **values,
    ):
        """Generate lazy random manifold data."""
        if isinstance(arg_names, str):
            arg_names = (arg_names,)

        point_names = [name for name in arg_names if "point" in name]
        tangent_names = [name for name in arg_names if name.startswith("tangent_vec")]

        ignored = set(arg_names) - set(point_names) - set(tangent_names)
        if ignored:
            raise ValueError(f"Unsupported argument names: {sorted(ignored)}.")

        if dependencies is None:
            dependencies = {}

            if len(point_names) == 1:
                point_name = point_names[0]
                dependencies.update({name: point_name for name in tangent_names})

        data = []
        for n_points in self.RANDOM_POINT_COUNTS:
            if exclude_single and n_points == 1:
                continue

            datum = dict(values)

            for arg_name in arg_names:
                if arg_name in dependencies:
                    base = datum[dependencies[arg_name]]
                    datum[arg_name] = LazyValue(
                        lambda base: self.data_generator.random_tangent_vec(base),
                        base,
                        label=arg_name,
                    )
                else:
                    datum[arg_name] = LazyValue(
                        lambda n=n_points: self.data_generator.random_point(n),
                        label=f"n_points={n_points}",
                    )

            data.append(datum)

        return data
