import collections
import inspect
import itertools
from functools import reduce


def unnest_list(ls):
    return list(itertools.chain(*ls))


def unnest(ls):
    if not is_non_string_iterable(ls):
        return [ls]

    data = []
    for datum_ in ls:
        data.extend(unnest(datum_))

    return data


def is_non_string_iterable(obj):
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str)


def params_to_kwargs(obj, ignore=(), renamings=None, ignore_private=False, func=None):
    """Get dict with selected object attributes.

    Parameters
    ----------
    obj : object
        Object with desired attributes.
    ignore : tuple[str]
        Attributes to ignore.
    renamings: dict
        Attribute renamings.
    ignore_private: bool
        Whether to ignore private attributes.
    func : callable
        Function to get signature from. Attributes
        not in the signature are ignored.

    Returns
    -------
    kwargs : dict
    """
    kwargs = obj.__dict__.copy()

    if func is not None:
        params = inspect.signature(func).parameters
        ignore = list(ignore) + [key for key in kwargs if key not in params]

    if ignore:
        for key in ignore:
            kwargs.pop(key)

    if renamings is not None:
        for old_key, new_key in renamings.items():
            kwargs[new_key] = kwargs.pop(old_key)

    if ignore_private:
        private_keys = list(filter(lambda key: key.startswith("_"), kwargs.keys()))
        for key in private_keys:
            kwargs.pop(key)

    return kwargs


def unnest_dict(nested_dict, sep="/", current_key="", flat_dict=None):
    sep_ = sep if current_key else ""

    if flat_dict is None:
        flat_dict = {}

    for key, value in nested_dict.items():
        new_key = f"{current_key}{sep_}{key}"

        if not isinstance(value, dict):
            flat_dict[new_key] = value
        else:
            flat_dict = unnest_dict(
                value, sep=sep, current_key=new_key, flat_dict=flat_dict
            )

    return flat_dict


def nest_dict_outer_level(flat_dict, sep="/"):
    nested_dict = {}

    for key, value in flat_dict.items():
        outer_key, inner_key = key.split(sep, maxsplit=1)

        inner_dict = nested_dict[outer_key] = nested_dict.get(outer_key, {})
        inner_dict[inner_key] = value

    return nested_dict


def nest_dict_inner_level(flat_dict, sep="/"):
    nested_dict = {}

    for key, value in flat_dict.items():
        outer_key, inner_key = key.rsplit(sep, maxsplit=1)

        inner_dict = nested_dict[outer_key] = nested_dict.get(outer_key, {})
        inner_dict[inner_key] = value

    return nested_dict


def nest_dict(flat_dict, sep="/"):
    while True:
        try:
            flat_dict = nest_dict_inner_level(flat_dict, sep=sep)
        except ValueError:
            # when unpack error is raised
            break

    return flat_dict


def compose_all(*funcs):
    # e.g. f = compose_all(str, abs, int)   # f(x) = str(abs(int(x)))
    funcs = list(filter(lambda x: x is not None, funcs))
    if len(funcs) == 1:
        return funcs[0]

    return reduce(lambda f, g: lambda x: f(g(x)), funcs)


def closest(lst, value):
    # https://www.geeksforgeeks.org/python/python-find-closest-number-to-k-in-given-list/
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - value))]
