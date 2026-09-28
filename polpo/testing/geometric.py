import inspect
import itertools
import operator
import random

import pytest
from geomstats.test.random import get_data_generator
from geomstats.vectorization import repeat_point

from .composition import CompositeGeometricCaseData
from .data import CaseData, TestDatum
from .lazy import LazyValue, materialize_lazy_values


class BaseGeometricCaseData(CaseData):
    def __init__(
        self,
        point_counts=None,
        time_counts=None,
        excluded_methods=(),
    ):
        super().__init__(excluded_methods=excluded_methods)

        self.point_counts = (
            [1] + random.sample(range(2, 5), 1)
            if point_counts is None
            else point_counts
        )
        # TODO: might not be a ManifoldTestData attribute
        self.time_counts = (
            [1] + random.sample(range(2, 5), 1) if time_counts is None else time_counts
        )

    def __add__(self, other):
        return CompositeGeometricCaseData(self, other)

    def get_decorators(self):
        return [materialize_lazy_values]

    def get_data_methods(self):
        return self._get_data_methods(
            excluded_suffixes="_vec_test_data",
        )

    def get_vectorization_data_methods(self):
        return self._get_data_methods(
            suffix="_vec_test_data",
        )

    def _get_data_generator(self, data_space=None):
        raise NotImplementedError

    def _prepare_generation(self, arg_names, dependencies=None, data_space=None):
        data_generator = self._get_data_generator(data_space)
        arg_names, dependencies = self._resolve_arg_names(
            arg_names,
            dependencies,
        )

        return arg_names, dependencies, data_generator

    def _resolve_arg_names(self, arg_names, dependencies=None):
        if isinstance(arg_names, str):
            arg_names = (arg_names,)

        point_names = [name for name in arg_names if "point" in name]
        tangent_names = [name for name in arg_names if _is_tangent_arg(name)]

        ignored = set(arg_names) - set(point_names) - set(tangent_names)
        if ignored:
            raise ValueError(f"Unsupported argument names: {sorted(ignored)}.")

        if dependencies is None:
            dependencies = {}

            if len(point_names) == 1:
                point_name = point_names[0]
                dependencies.update({name: point_name for name in tangent_names})

        return arg_names, dependencies

    def _generate_random_data(
        self,
        arg_names,
        dependencies,
        data_generator,
        exclude_single=False,
        **values,
    ):
        data = []

        for n_points in self.point_counts:
            if exclude_single and n_points == 1:
                continue

            datum = self._generate_random_datum(
                arg_names,
                dependencies,
                data_generator,
                n_points=n_points,
                **values,
            )

            data.append(TestDatum(datum, marks=(pytest.mark.random,)))

        return data

    def _generate_random_datum(
        self,
        arg_names,
        dependencies,
        data_generator,
        n_points=1,
        **values,
    ):
        datum = dict(values)

        for arg_name in arg_names:
            if arg_name in dependencies:
                datum[arg_name] = LazyValue(
                    _random_tangent_vec,
                    data_generator,
                    datum[dependencies[arg_name]],
                    label=f"n={n_points}",
                )
            else:
                datum[arg_name] = LazyValue(
                    _random_point,
                    data_generator,
                    n_points,
                    label=f"n={n_points}",
                )

        return datum

    def generate_random_data(
        self,
        arg_names,
        data_space=None,
        dependencies=None,
        exclude_single=False,
        **values,
    ):
        arg_names, dependencies, data_generator = self._prepare_generation(
            arg_names,
            dependencies,
            data_space,
        )

        return self._generate_random_data(
            arg_names,
            dependencies,
            data_generator,
            exclude_single=exclude_single,
            **values,
        )

    def _generate_vectorization_data(
        self,
        arg_names,
        dependencies,
        data_generator,
        expected_func,
        expected_name="expected",
        vectorization_type=None,
        n_reps=2,
        **values,
    ):
        vectorization_type = _resolve_vectorization_type(
            arg_names,
            dependencies,
            vectorization_type,
        )

        datum = self._generate_random_datum(
            arg_names,
            dependencies,
            data_generator,
            n_points=1,
        )

        expected_value = LazyValue(
            expected_func,
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

        if isinstance(expected_name, str):
            expected_values = {
                expected_name: expected_value,
            }
        else:
            expected_values = {
                name: LazyValue(
                    operator.itemgetter(index),
                    expected_value,
                )
                for index, name in enumerate(expected_name)
            }

        expected_values = {
            name: LazyValue(
                repeat_point,
                value,
                n_reps=n_reps,
                label=f"r={n_reps}",
            )
            for name, value in expected_values.items()
        }

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
                        label=f"r={n_reps}",
                    )

            new_datum.update(expected_values)

            data.append(new_datum)

        return data

    def generate_vectorization_data(
        self,
        arg_names,
        data_space=None,
        op_name=None,
        expected_name="expected",
        vectorization_type=None,
        dependencies=None,
        op_evaluator=None,
        n_reps=2,
        op_target=None,
        **values,
    ):
        if op_name is None:
            op_name = _get_op_name_from_caller()

        arg_names, dependencies, data_generator = self._prepare_generation(
            arg_names,
            dependencies,
            # TODO: space to data_space
            data_space,
        )

        if op_evaluator is None:
            op_evaluator = lambda op, **kwargs: op(**kwargs)

        arg_names, dependencies = self._resolve_arg_names(
            arg_names,
            dependencies,
        )

        expected_func = lambda **kwargs: op_evaluator(
            self._get_operation(op_name, op_target),
            **kwargs,
        )

        return self._generate_vectorization_data(
            arg_names,
            dependencies,
            data_generator,
            expected_func,
            expected_name=expected_name,
            vectorization_type=vectorization_type,
            n_reps=n_reps,
            **values,
        )


class GeometricCaseData(BaseGeometricCaseData):
    def __init__(
        self,
        space=None,
        point_counts=None,
        time_counts=None,
        excluded_methods=(),
    ):
        super().__init__(
            point_counts=point_counts,
            time_counts=time_counts,
            excluded_methods=excluded_methods,
        )

        self.space = space
        self.data_generator = LazyValue(lambda: get_data_generator(self.space))

    def _get_data_generator(self, data_space=None):
        if data_space is not None:
            raise ValueError(f"Unknown space {data_space!r}.")

        return self.data_generator

    def _get_operation(self, op_name, op_target="space"):
        if op_target is None or op_target == "space":
            target = self.space

        elif op_target == "metric":
            target = self.space.metric

        else:
            raise ValueError(f"Unknown operation target {op_target!r}.")

        return getattr(target, op_name)

    def generate_random_data(
        self,
        arg_names,
        dependencies=None,
        exclude_single=False,
        **values,
    ):
        return super().generate_random_data(
            arg_names,
            dependencies=dependencies,
            exclude_single=exclude_single,
            **values,
        )

    def generate_vectorization_data(
        self,
        arg_names,
        op_name=None,
        expected_name="expected",
        vectorization_type=None,
        dependencies=None,
        op_evaluator=None,
        n_reps=2,
        on_metric=False,
        **values,
    ):
        return super().generate_vectorization_data(
            arg_names,
            data_space=None,
            op_name=op_name,
            expected_name=expected_name,
            vectorization_type=vectorization_type,
            dependencies=dependencies,
            op_evaluator=op_evaluator,
            n_reps=n_reps,
            op_target="metric" if on_metric else "space",
            **values,
        )


class FiberBundleCaseData(BaseGeometricCaseData):
    def __init__(
        self,
        total_space=None,
        base_space=None,
        point_counts=None,
        time_counts=None,
        excluded_methods=(),
    ):
        super().__init__(
            point_counts=point_counts,
            time_counts=time_counts,
            excluded_methods=excluded_methods,
        )

        self.total_space = total_space
        self.base_space = base_space

        self.total_space_data_generator = LazyValue(
            lambda: get_data_generator(self.total_space)
        )

        self.base_space_data_generator = LazyValue(
            lambda: get_data_generator(self.base_space)
        )

    def _get_data_generator(self, data_space=None):
        if data_space is None or data_space == "total":
            return self.total_space_data_generator

        if data_space == "base":
            return self.base_space_data_generator

        raise ValueError(f"Unknown ``space`` {data_space}")

    def _get_operation(self, op_name, op_target="bundle"):
        if op_target is None or op_target == "bundle":
            target = self.total_space.fiber_bundle

        else:
            raise ValueError(f"Unknown operation target {op_target!r}.")

        return getattr(target, op_name)

    def _generate_lifted_random_datum(
        self,
        point_name,
        horizontal_names=(),
        tangent_names=(),
        n_points=1,
        **values,
    ):
        def _random_horizontal_vec(data_generator, base_point, fiber_point):
            tangent_vec = data_generator.random_tangent_vec(base_point)

            return self.total_space.fiber_bundle.horizontal_lift(
                tangent_vec,
                fiber_point=fiber_point,
                base_point=base_point,
            )

        datum = dict(values)

        base_space_point = LazyValue(
            _random_point,
            self.base_space_data_generator,
            n_points,
            label=f"n={n_points}",
        )
        fiber_point = LazyValue(
            lambda point: self.total_space.fiber_bundle.lift(point),
            base_space_point,
            label=f"n={n_points}",
        )

        datum[point_name] = fiber_point

        for name in horizontal_names:
            datum[name] = LazyValue(
                _random_horizontal_vec,
                self.base_space_data_generator,
                base_space_point,
                fiber_point,
                label=f"n={n_points}",
            )

        for name in tangent_names:
            datum[name] = LazyValue(
                _random_tangent_vec,
                self.total_space_data_generator,
                fiber_point,
                label=f"n={n_points}",
            )

        return datum


class GeometricMapCaseData(BaseGeometricCaseData):
    def __init__(
        self,
        space=None,
        image_space=None,
        point_counts=None,
        time_counts=None,
        excluded_methods=(),
    ):
        super().__init__(
            point_counts=point_counts,
            time_counts=time_counts,
            excluded_methods=excluded_methods,
        )

        self.space = space
        self.image_space = image_space

        self.data_generator = LazyValue(lambda: get_data_generator(self.space))
        self.image_data_generator = LazyValue(
            lambda: get_data_generator(self.image_space)
        )

    def _get_data_generator(self, data_space=None):
        if data_space is None or data_space == "space":
            return self.data_generator

        if data_space == "image":
            return self.image_space_data_generator

        raise ValueError(f"Unknown space {data_space!r}.")

    def _get_operation(self, op_name, op_target="space"):
        if op_target is None or op_target == "space":
            target = self.space

        elif op_target == "metric":
            target = self.space.metric

        else:
            raise ValueError(f"Unknown operation target {op_target!r}.")

        return getattr(target, op_name)


def _random_point(data_generator, n_points):
    return data_generator.random_point(n_points)


def _random_tangent_vec(data_generator, base_point):
    return data_generator.random_tangent_vec(base_point)


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


def _resolve_vectorization_type(
    arg_names,
    dependencies,
    vectorization_type=None,
):
    if vectorization_type is not None:
        return vectorization_type

    if not dependencies:
        return "sym"

    dependency_names = set(dependencies.values())

    repeat_indices = [
        index for index, name in enumerate(arg_names) if name not in dependency_names
    ]

    if len(repeat_indices) == len(arg_names):
        return "sym"

    return "repeat-" + "-".join(map(str, repeat_indices))


def _is_tangent_arg(name):
    return name.startswith(
        ("tangent_vec", "vector", "vec", "initial_tangent_vec", "direction")
    )


def _get_op_name_from_caller():
    suffix = "_vec_test_data"

    frame = inspect.currentframe().f_back
    while frame is not None:
        caller_name = frame.f_code.co_name
        if caller_name.endswith(suffix):
            return caller_name.removesuffix(suffix)

        frame = frame.f_back

    raise ValueError("Cannot infer operation name from caller.")
