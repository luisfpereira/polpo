import numpy as np

import polpo.preprocessing.dict as ppdict
from polpo.preprocessing import (
    IfCondition,
    IndexMap,
    Map,
    NestingSwapper,
    Pipeline,
    WrapInList,
)


class DictsToXY(Pipeline):
    """Convert two key-matching dictionaries into (X, y) for learning.

    Expects a list of two dictionaries with identical keys.
    Merges them by key, swaps the nesting to produce (key -> [x, y]) pairs,
    and returns a tuple of two elements:
    an array for X and a list for y.
    """

    def __init__(self):
        # TODO: generalize for y other than meshes
        # TODO: can be generalized for more cases
        x_step = IfCondition(
            step=ppdict.DictToValuesList(),
            else_step=WrapInList(),
            condition=lambda x: isinstance(x, dict),
        )

        super().__init__(
            steps=[
                ppdict.DictMerger(),
                NestingSwapper(),
                IndexMap(index=0, step=Map(x_step) + (lambda x: np.asarray(x))),
            ]
        )


class NestedDictsToXY(Pipeline):
    """Convert two key-matching dictionaries into (X, y) for learning.

    Similar to ``DictsToXY``, but assumes ``y`` is a nested ``dict``
    (outer keys correspond to different outputs; merging is done using inner keys).
    """

    def __init__(self):
        x_step = IfCondition(
            step=Map(step=ppdict.DictToValuesList()),
            else_step=lambda x: np.asarray(x)[:, None],
            condition=lambda x: isinstance(x[0], dict),
        )

        super().__init__(
            steps=[
                IndexMap(index=1, step=ppdict.NestedDictSwapper()),
                ppdict.DictMerger(),
                NestingSwapper(),
                IndexMap(index=0, step=x_step),
                IndexMap(index=1, step=ppdict.ListDictSwapper()),
            ]
        )
