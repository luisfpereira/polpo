"""Preprocessing steps involving strings."""

import re

from ._preprocessing import (  # noqa:F401
    Contains,
    ContainsAll,
    ContainsAny,
    ExceptionToWarning,
    MethodApplier,
)
from .base import PreprocessingStep


class DigitFinder(PreprocessingStep):
    """Find digits in a strings.

    Parameters
    ----------
    index : int
        Which index to pick in case of multiple
        digit sets are expected to be found.
    """

    def __init__(self, index=-1):
        super().__init__()
        self.index = index

        self._pattern = re.compile(r"\d+")

    def __call__(self, string):
        """Apply step.

        Parameters
        ----------
        string : str

        Returns
        -------
        int
        """
        digits = self._pattern.findall(string)
        return int(digits[self.index])


class RegexGroupFinder(PreprocessingStep):
    def __init__(self, pattern):
        # pattern must contain a group
        super().__init__()
        self._pattern = re.compile(pattern)

    def __call__(self, data):
        return self._pattern.search(data).group(1)


class StartsWith(MethodApplier):
    def __init__(self, value):
        super().__init__(value, method="startswith")


class EndsWith(MethodApplier):
    def __init__(self, value):
        super().__init__(value, method="endswith")


class EndsWithAny(PreprocessingStep):
    # TODO: generalize based on ContainsAny?
    def __init__(self, items, negate=False):
        super().__init__()
        self.items = items
        self.negate = negate

    def __call__(self, data):
        """Apply step.

        Returns
        -------
        membership : bool
            Membership or lack of it (depending on negate) for all items.
        """
        out = any(data.endswith(item) for item in self.items)
        if self.negate:
            return not out

        return out


class TryToInt(ExceptionToWarning):
    def __init__(self, warn=False):
        super().__init__(int, warn=warn)
