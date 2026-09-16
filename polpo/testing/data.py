import functools
import itertools
import random
from collections.abc import Mapping
from dataclasses import dataclass

import pytest
from geomstats.test.random import get_data_generator
from geomstats.vectorization import repeat_point

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
        arg_names, dependencies = self._resolve_arg_names(arg_names, dependencies)

        data = []
        for n_points in self.RANDOM_POINT_COUNTS:
            if exclude_single and n_points == 1:
                continue

            datum = self._generate_random_datum(
                arg_names, dependencies, n_points=n_points, **values
            )

            data.append(TestDatum(datum, marks=(pytest.mark.random,)))

        return data

    def _resolve_arg_names(self, arg_names, dependencies=None):
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

        return arg_names

    def _generate_random_datum(self, arg_names, dependencies, n_points=1, **values):
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

    def generate_vectorization_data(
        self,
        arg_names,
        op_name,
        expected_name="expected",
        vectorization_type="sym",
        dependencies=None,
        n_reps=2,
        on_metric=True,
        **values,
    ):
        arg_names, dependencies = self._resolve_arg_names(arg_names, dependencies)

        datum = self._generate_random_datum(
            arg_names,
            dependencies,
            n_points=1,
        )

        expected_value = LazyValue(
            lambda **kwargs: getattr(
                self.space.metric if on_metric else self.space,
                op_name,
            )(**kwargs),
            **datum,
        )

        return self._vectorize_datum(
            datum,
            expected_value,
            vectorization_type,
            expected_name,
            n_reps,
            **values,
        )

    def _vectorize_datum(
        self,
        datum,
        expected_value,
        vectorization_type,
        expected_name="expected",
        n_reps=2,
        **values,
    ):
        arg_names = list(datum)
        combinations = _get_vectorization_combinations(
            len(arg_names),
            vectorization_type,
        )

        expected_value_rep = LazyValue(
            repeat_point,
            expected_value,
            n_reps=n_reps,
        )

        data = []
        for combination in combinations:
            new_datum = {**values, **datum}

            for arg_name, repeat in zip(arg_names, combination):
                if repeat:
                    # TODO: add batch shape as label?
                    new_datum[arg_name] = LazyValue(
                        repeat_point,
                        datum[arg_name],
                        n_reps=n_reps,
                        expand=True,
                    )

            new_datum[expected_name] = expected_value_rep

            data.append(new_datum)

        return data


def _get_vectorization_combinations(n_args, vectorization_type):
    """Get repetition combinations for vectorization tests.

    Parameters
    ----------
    n_args : int
        Number of input arguments that can be vectorized.
    vectorization_type : str
        Strategy used to generate repetition combinations.

        Supported values are:

        * ``"basic"``: repeat all arguments.
        * ``"sym"``: generate every non-empty repetition combination.
        * ``"repeat-i-j-..."``: generate non-empty repetition combinations
          involving only the specified argument indices, together with the
          combination in which all arguments are repeated.

    Returns
    -------
    combinations : list of tuple of int
        Repetition combinations. Each tuple has length ``n_args`` and contains
        zeros and ones, where ``1`` indicates that the corresponding argument
        should be repeated.

    Raises
    ------
    ValueError
        If ``vectorization_type`` is unknown or contains invalid argument
        indices.

    Examples
    --------
    For three arguments, ``"repeat-0-2"`` produces combinations equivalent to
    ``001``, ``100``, ``101``, and ``111``.
    """
    if vectorization_type == "basic":
        return [(1,) * n_args]

    combinations = list(itertools.product((0, 1), repeat=n_args))
    combinations.remove((0,) * n_args)

    if vectorization_type == "sym" or n_args == 1:
        return combinations

    if not vectorization_type.startswith("repeat-"):
        raise ValueError(f"Unknown vectorization type: {vectorization_type!r}.")

    try:
        repeat_indices = {
            int(index)
            for index in vectorization_type.removeprefix("repeat-").split("-")
        }
    except ValueError as error:
        raise ValueError(
            f"Unable to understand vectorization type {vectorization_type!r}."
        ) from error

    if not repeat_indices or not repeat_indices < set(range(n_args)):
        raise ValueError(
            f"Invalid repetition indices for {n_args} arguments: "
            f"{sorted(repeat_indices)}."
        )

    if len(repeat_indices) == n_args:
        return combinations

    return [
        combination
        for combination in combinations
        if (
            all(
                not combination[index]
                for index in range(n_args)
                if index not in repeat_indices
            )
            or all(combination)
        )
    ]
