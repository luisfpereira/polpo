import inspect
import types

import pytest


def _collect_members(attrs, bases, predicate):
    """Collect class members matching a predicate.

    Members are collected from the base classes and from the attributes of the
    class being created. Attributes defined on the new class override inherited
    members with the same name. Among base classes, earlier bases take
    precedence over later ones.

    Parameters
    ----------
    attrs : dict
        Attributes defined on the class being created.
    bases : tuple of type
        Base classes of the class being created.
    predicate : callable
        Function taking a member name and returning whether that member should
        be collected.

    Returns
    -------
    dict
        Mapping from member names to matching members.
    """
    members = {}

    for base in reversed(bases):
        members.update(
            (name, getattr(base, name)) for name in dir(base) if predicate(name)
        )

    members.update((name, attr) for name, attr in attrs.items() if predicate(name))

    return members


def _get_pytest_marks(func):
    if not hasattr(func, "pytestmark"):
        return []

    return [elem for elem in func.pytestmark if isinstance(elem, pytest.Mark)]


def _copy_func(
    f,
    name=None,
):
    """Copy function.

    Return a function with same code, globals, defaults, closure, and
    name (or provide a new name).

    Additionally, keyword arguments are transformed into positional arguments for
    compatibility with pytest.
    """
    fn = types.FunctionType(
        f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__
    )
    fn.__dict__.update(f.__dict__)

    sign = inspect.signature(fn)
    defaults, new_params = {}, []
    for param in sign.parameters.values():
        if param.default is inspect._empty:
            new_params.append(param)
        else:
            new_params.append(inspect.Parameter(param.name, kind=1))
            defaults[param.name] = param.default
    new_sign = sign.replace(parameters=new_params)
    fn.__signature__ = new_sign

    return fn, defaults
