"""Preprocessing steps involving files."""

import os
import warnings

from ._preprocessing import MethodApplier
from .base import PreprocessingStep


class IsFileType(MethodApplier):
    """Check extension of file.

    Parameters
    ----------
    ext : str
        Extension.
    """

    def __init__(self, ext):
        super().__init__(f".{ext}", method="endswith")


class FileFinder(PreprocessingStep):
    """Find files given rules.

    Parameters
    ----------
    data_dir : str
        Searching directory.
    rules : callable or list[callable]
        Rules to filter files with.
    warn : bool
        Whether to warn if can't find file.
    as_list : bool
        Whether to return a list if only single element.
    """

    def __init__(self, data_dir=None, rules=(), warn=True, as_list=False):
        super().__init__()
        self.data_dir = data_dir
        self.warn = warn
        self.as_list = as_list

        if callable(rules):
            rules = [rules]

        self.rules = rules

    def __call__(self, data=None):
        """Apply step.

        Parameters
        ----------
        data_dir : str
            Searching directory.

        Returns
        -------
        list[str] or str
        """
        data_dir = ExpandUser()(data or self.data_dir)

        if not os.path.exists(data_dir):
            if self.warn:
                warnings.warn(f"`{data_dir}` does not exist.")
            return []

        files = os.listdir(data_dir)

        for rule in self.rules:
            files = filter(rule, files)

        out = list(map(lambda name: data_dir / name, files))

        if self.warn and len(out) == 0:
            warnings.warn(f"Couldn't find file in: {data_dir}")

        if len(out) == 1 and not self.as_list:
            return out[0]

        return out


class PathShortener(PreprocessingStep):
    """Shorten path.

    Parameters
    ----------
    init_index : int
        Initial index for concatenation.
    last_index : int
        Last index for concatenation.
    """

    def __init__(self, init_index=-1, last_index=None):
        self.init_index = init_index
        self.last_index = last_index

    def __call__(self, path):
        """Apply step.

        Parameters
        ----------
        path : str
            Path name.

        Returns
        -------
        str
        """
        path_ = path if isinstance(path, str) else path.as_posix()
        path_ls = path_.split(os.path.sep)
        return f"{os.path.sep}".join(path_ls[self.init_index : self.last_index])


class ExpandUser(PreprocessingStep):
    def __call__(self, filename):
        if isinstance(filename, str):
            if "~" in filename:
                return os.path.expanduser(filename)
        else:
            return filename.expanduser()

        return filename
