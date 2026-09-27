import functools

from .execution import get_execution_key


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
