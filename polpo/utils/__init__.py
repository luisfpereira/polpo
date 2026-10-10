# NB: do not explictly import anything that depends on third part library
import collections
import getpass
import glob
import importlib
import inspect
import itertools
import socket
import string
from pathlib import Path

from .dict_ import *  # noqa: F403
from .pd import *  # noqa: F403

try:
    from .url import *  # noqa: F403
except ImportError:
    pass


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


def as_list(data):
    if isinstance(data, list):
        return data

    if is_non_string_iterable(data):
        return list(data)

    return [data]


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


def custom_order(reference):
    # behavior is random if element is not in reference
    order_ = {val: index for index, val in enumerate(reference)}
    n_reference = len(order_)

    def _custom_order(x):
        return order_.get(x, n_reference)

    return _custom_order


def plot_shape_from_n_plots(n_plots, n_axis=2, axis=1):
    # TODO: compute space filler?
    n_axis_0 = min(n_axis, n_plots)
    n_axis_1 = (n_plots + n_axis_0 - 1) // n_axis_0

    if axis == 1:
        return n_axis_1, n_axis_0

    return n_axis_0, n_axis_1


def plot_index_to_shape(index, n_axis, rowise=False):
    # TODO: find better name
    a, b = index // n_axis, index % n_axis

    if rowise:
        return b, a

    return a, b


def get_first(data):
    if isinstance(data, dict):
        return next(iter(data.values()))

    return data[0]


def in_frank():
    return socket.gethostname() == "frank"


def get_frank_user_scratch():
    return Path(f"/scratch/{getpass.getuser()}")


def expand_path_names(names):
    out = []
    for name in names:
        if any(ch in name for ch in "*?[]"):
            out.extend(Path(p) for p in glob.glob(name, recursive=True))
        else:
            out.append(Path(name))

    seen = set()
    uniq = []
    for path in out:
        rp = path.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(path)
    return uniq


def get_results_path():
    if in_frank():
        return get_frank_user_scratch()

    return Path.home() / ".polpo/results"


def has_package(package_name):
    """Check if package is installed.

    Parameters
    ----------
    package_name : str
        Package name.
    """
    return importlib.util.find_spec(package_name) is not None


def index_to_letters(index):
    result = ""

    while True:
        index, remainder = divmod(index, 26)
        result = string.ascii_uppercase[remainder] + result

        if index == 0:
            return result

        index -= 1
