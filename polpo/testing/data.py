from collections.abc import Mapping
from dataclasses import dataclass

from .lazy import materialize_lazy_values


@dataclass(frozen=True)
class TestDatum:
    data: object
    marks: tuple = ()


def _unpack_test_datum(datum):
    if isinstance(datum, TestDatum):
        return datum.data, datum.marks

    return datum, ()


def _normalize_datum(datum, arg_names, func_name=None):
    if isinstance(datum, Mapping):
        unknown = datum.keys() - set(arg_names)
        if unknown:
            message = f"Unknown argument names: {sorted(unknown)}."
            if func_name is not None:
                message = f"{func_name}: {message}"

            raise ValueError(message)

        return datum

    if len(datum) > len(arg_names):
        message = f"Got {len(datum)} values for {len(arg_names)} arguments."
        if func_name is not None:
            message = f"{func_name}: {message}"
        raise ValueError(message)

    return dict(zip(arg_names, datum))


class CaseData:
    def __init__(self, excluded_methods=()):
        self.excluded_methods = set(excluded_methods)

    def _get_data_methods(self, suffix="_test_data", excluded_suffixes=()):
        if isinstance(excluded_suffixes, str):
            excluded_suffixes = (excluded_suffixes,)

        methods = {}

        for method_name in dir(self):
            if not method_name.endswith(suffix):
                continue

            if any(method_name.endswith(suffix_) for suffix_ in excluded_suffixes):
                continue

            name = method_name.removesuffix(suffix)

            if name in self.excluded_methods:
                continue

            methods[name] = getattr(self, method_name)

        return methods

    def get_data_methods(self):
        return self._get_data_methods()

    def get_decorators(self):
        return ()

    def _with_values(self, datum, **values):
        if isinstance(datum, TestDatum):
            return TestDatum(
                {**datum.data, **values},
                marks=datum.marks,
            )

        return {**datum, **values}

    def with_values(self, data, **values):
        return [self._with_values(datum, **values) for datum in data]


class LazyCaseData(CaseData):
    def get_decorators(self):
        return [materialize_lazy_values]
