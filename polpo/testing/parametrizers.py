import os

import nbformat
import pytest

# TODO: harmonize with geomstats
from geomstats.test.parametrizers import (
    _exec_notebook,
    _is_test,
    _raise_missing_testing_data,
    _raise_missing_tests,
)

from ._utils import _collect_members
from .core import TestFunction


class NotebooksParametrizer(type):
    def __new__(cls, name, bases, attrs):
        def _create_new_test(path, **kwargs):
            def new_test(self):
                return _exec_notebook(path=path)

            return new_test

        testing_data = locals()["attrs"].get("testing_data")
        _raise_missing_testing_data(testing_data)

        paths = testing_data.paths

        for path in paths:
            name = path.split(os.sep)[-1].split(".")[0]

            func_name = f"test_{name}"
            test_func = _create_new_test(path)

            metadata = nbformat.read(path, as_version=4).metadata

            for marker_ in metadata.get("markers", []):
                marker = getattr(pytest.mark, marker_)
                test_func = marker()(test_func)

            attrs[func_name] = test_func

        return super().__new__(cls, name, bases, attrs)


class DataBasedParametrizer(type):
    """Metaclass for test classes driven by data definition.

    It differs from `Parametrizer` because every test data function must have
    an associated test function, instead of the opposite.
    """

    def __new__(cls, name, bases, attrs):
        testing_data = locals()["attrs"].get("testing_data")
        _raise_missing_testing_data(testing_data)

        test_fncs = _collect_members(attrs, bases, _is_test)
        data_fncs = testing_data.get_data_methods()

        tests = {}
        for name, func in test_fncs.items():
            # TODO: handle vec; something like expand?
            # TODO: add that already here?

            tests[name] = TestFunction(name, func).set_data_method(
                data_fncs.pop(name.removeprefix("test_"), None)
            )

        _raise_missing_tests(data_fncs)

        decorators = testing_data.get_decorators()
        for name, func in tests.items():
            tests[name] = func.build(decorators=decorators)

        attrs.update(tests)

        return super().__new__(cls, name, bases, attrs)
