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


def unnest_dict(nested_dict, sep="/", current_key=None, flat_dict=None):
    """Flatten a nested dictionary by combining keys along each path.

    Parameters
    ----------
    nested_dict : dict
        Dictionary to flatten.
    sep : str or None
        Separator used to join keys. If not a string, keys are
        represented as tuples.
    current_key : str or tuple
        Key prefix accumulated during recursion.
    flat_dict : dict
        Dictionary populated during recursion.

    Returns
    -------
    flat_dict : dict
        Flattened dictionary mapping key paths to leaf values.
    """
    if isinstance(sep, str):
        prefix = "" if current_key is None else f"{current_key}{sep}"

        def _update_key(key):
            return f"{prefix}{key}"

    else:
        prefix = () if current_key is None else current_key

        def _update_key(key):
            return prefix + (key,)

    if flat_dict is None:
        flat_dict = {}

    for key, value in nested_dict.items():
        new_key = _update_key(key)

        if not isinstance(value, dict):
            flat_dict[new_key] = value
        else:
            flat_dict = unnest_dict(
                value, sep=sep, current_key=new_key, flat_dict=flat_dict
            )

    return flat_dict


def _nest_dict_inner_level(flat_dict, sep="/"):
    """Move the final segment of each key into an inner ``dict``.

    For string keys, segments are separated by ``sep``.

    Parameters
    ----------
    flat_dict : dict
        Dictionary with composite keys.
    sep : str
        Separator used to split string keys. Tuple keys are
        unpacked directly.

    Returns
    -------
    nested_dict : dict
        Dictionary with the final key component moved to an
        inner dictionary.
    """
    nested_dict = {}

    for key, value in flat_dict.items():
        if isinstance(key, str):
            outer_key, inner_key = key.rsplit(sep, maxsplit=1)
        else:
            outer_key, inner_key = key

        inner_dict = nested_dict[outer_key] = nested_dict.get(outer_key, {})
        inner_dict[inner_key] = value

    return nested_dict


def nest_dict(flat_dict, sep="/"):
    """Convert a flat dictionary into a nested dictionary.

    Repeatedly nests the final component of each key until no
    further nesting is possible.

    Parameters
    ----------
    flat_dict : dict
        Dictionary with composite keys, typically strings separated
        by `sep`.
    sep : str
        Separator used to split string keys.

    Returns
    -------
    nested_dict : dict
        Dictionary reconstructed from the composite keys.

    Notes
    -----
    Keys are expected to have a consistent nesting depth.
    """
    while flat_dict:
        try:
            flat_dict = _nest_dict_inner_level(flat_dict, sep=sep)
        except ValueError:
            # when unpack error is raised
            break

    return flat_dict


def merge_dicts(dicts, *, check_duplicates=False):
    """Merge a sequence of ``dict``.

    Parameters
    ----------
    dicts : iterable of dict
        ``dict`` to merge, in order.
    check_duplicates : bool
        Whether to raise an error when a key occurs in multiple
        ``dict``. Otherwise, later values overwrite earlier ones.

    Returns
    -------
    merged : dict
        Dictionary containing the merged key-value pairs.

    Raises
    ------
    ValueError
        If duplicate keys are found and `check_duplicates` is True.
    """
    result = {}

    for dict_ in dicts:
        if check_duplicates:
            duplicates = result.keys() & dict_.keys()
            if duplicates:
                raise ValueError(f"Duplicate keys: {sorted(duplicates)}")

        result.update(dict_)

    return result
