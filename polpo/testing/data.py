from collections.abc import Mapping
from dataclasses import dataclass

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


class LazyValue:
    def __init__(self, factory):
        self.factory = factory

    def materialize(self):
        return self.factory()


class CachedLazyValue(LazyValue):
    def __init__(self, factory):
        super().__init__(factory)

        self._values = {}
        self._materialized = False

    def materialize(self):
        key = get_execution_key()

        if key not in self._values:
            self._values[key] = super().materialize()

        return self._values[key]
