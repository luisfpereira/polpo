import getpass
import glob
import importlib
import inspect
import itertools
import socket
import string
from pathlib import Path


def unnest_list(ls):
    """Flatten a sequence of iterables by one level.

    Parameters
    ----------
    ls : iterable of iterables
        Nested elements to flatten.

    Returns
    -------
    flattened : list
        Elements concatenated in their original order.
    """
    return list(itertools.chain(*ls))


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


def in_frank():
    return socket.gethostname() == "frank"


def get_frank_user_scratch():
    return Path(f"/scratch/{getpass.getuser()}")


def expand_path_names(names):
    """Expand glob patterns into unique paths.

    Parameters
    ----------
    names : iterable of str
        File paths or glob patterns. Supports recursive patterns.

    Returns
    -------
    paths : list of pathlib.Path
        Expanded paths, preserving encounter order and removing
        duplicates based on resolved paths.

    Notes
    -----
    Literal paths are retained even if they do not exist.
    Unmatched glob patterns contribute no paths.
    """
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
    """Return the default Polpo results directory.

    Returns
    -------
    path : pathlib.Path
        Results directory under the user's home directory.
    """
    return Path.home() / ".polpo" / "results"


def has_package(package_name):
    """Check if package is installed.

    Parameters
    ----------
    package_name : str
        Package name.
    """
    return importlib.util.find_spec(package_name) is not None


def index_to_letters(index):
    """Convert a zero-based index to an uppercase alphabetic label.

    Labels follow spreadsheet-style ordering: A, B, ..., Z, AA, AB, ...

    Parameters
    ----------
    index : int
        Nonnegative zero-based index.

    Returns
    -------
    label : str
        Alphabetic representation of the index.
    """
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
            outer_key, inner_key = key[:-1], key[-1]
            if len(outer_key) == 1:
                outer_key = outer_key[0]

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
