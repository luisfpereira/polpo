import inspect

import pytest

from ._utils import _copy_func, _get_pytest_marks
from .data import _normalize_datum, _unpack_test_datum


class TestFunction:
    def __init__(self, name, func):
        self.name = name
        self.func = func

        self._test_marks = _get_pytest_marks(func)
        self._data_marks = []

        spec = inspect.getfullargspec(self.func)
        self.arg_names = spec.args[1:] + spec.kwonlyargs

        self.data_fnc = None

    @property
    def arg_str(self):
        return ", ".join(self.arg_names)

    def set_data_method(self, data_fnc):
        self.data_fnc = data_fnc
        self._data_marks = _get_pytest_marks(data_fnc)
        return self

    @property
    def active(self):
        return self.data_fnc is not None

    @property
    def skip(self):
        return pytest.mark.skip in self.marks

    @property
    def marks(self):
        return self._test_marks + self._data_marks

    def build(self, decorators=()):
        test_func, default_values = _copy_func(self.func)

        if self.skip:
            return pytest.mark.skip()(test_func)

        if not self.active:
            return pytest.mark.ignore()(test_func)

        for decorator in decorators:
            test_func = decorator(test_func)

        for mark in self._data_marks:
            test_func.pytestmark = [
                *getattr(test_func, "pytestmark", ()),
                *self._data_marks,
            ]

        # TODO: should still check return?
        # no args case
        if len(self.arg_names) == 0:
            return test_func

        data = []
        for datum in self.data_fnc():
            datum, datum_marks = _unpack_test_datum(datum)
            datum = {**default_values, **_normalize_datum(datum, self.arg_names)}

            values = [datum[name] for name in self.arg_names]
            data.append(pytest.param(*values, marks=datum_marks))

        return pytest.mark.parametrize(self.arg_str, data)(test_func)
