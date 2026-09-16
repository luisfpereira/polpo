import functools

from .data import LazyValue


def _materialize(value):
    if isinstance(value, LazyValue):
        return value.materialize()

    return value


def materialize_lazy_values(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = tuple(_materialize(value) for value in args)
        kwargs = {name: _materialize(value) for name, value in kwargs.items()}
        return func(*args, **kwargs)

    return wrapper
