import abc

import dash_bootstrap_components as dbc

from polpo.utils import compose_all, is_non_string_iterable


class Layout(abc.ABC):
    def __init__(self, sorter=None):
        # TODO: selector instead of sorter?
        if sorter is None:
            sorter = lambda x: x
        self.sorter = sorter

    @abc.abstractmethod
    def __call__(self, comps):
        pass


class DummyLayout(Layout):
    def __call__(self, comps):
        return self.sorter(comps)


class NestedLayout(Layout):
    def __init__(self, layouts, sorter=None):
        super().__init__(sorter=sorter)
        self.layouts = layouts

    def __call__(self, comps):
        comps = self.sorter(comps)
        for layout in self.layouts:
            comps = layout(comps)

        return comps


class StackInCard(Layout):
    def __init__(self, gap=3, sorter=None):
        super().__init__(sorter=sorter)
        self.gap = gap

    def __call__(self, comps):
        comps = self.sorter(comps)

        return [
            dbc.Card(
                [dbc.Stack(comps, gap=self.gap)],
                body=True,
            )
        ]


class GridLayout(Layout):
    """Grid layout.

    Parameters
    ----------
    width : None or str or int
        If int, [1, 12].
    as_stack : bool
        Whether to wrap rows with ``dbc.Stack```. Automatically ``True`` if ``row_gap``.
    row_gap : int
        Gap between rows.
    """

    def __init__(
        self,
        width=None,
        align="center",
        justify="center",
        sorter=None,
        as_stack=False,
        row_gap=0,
    ):
        super().__init__(sorter=sorter)

        if row_gap:
            as_stack = True

        self.align = align
        self.justify = justify
        self.width = width
        self.as_stack = as_stack
        self.row_gap = row_gap

    def __call__(self, row_comps):
        row_comps = self.sorter(row_comps)
        n_row_comps = len(row_comps)

        width_ls = self.width
        if not is_non_string_iterable(width_ls):
            width_ls = [width_ls] * n_row_comps

        if len(width_ls) != n_row_comps:
            raise ValueError(
                f"Must have has many row comps as width values: {n_row_comps} != {len(width_ls)}"
            )

        rows = []
        for col_comps, width in zip(row_comps, width_ls):
            if isinstance(col_comps, dbc.Row):
                rows.append(col_comps)
                continue

            if not isinstance(col_comps, (list, tuple)):
                col_comps = [col_comps]

            n_col_comps = len(col_comps)
            if not is_non_string_iterable(width):
                width = [width] * n_col_comps

            if len(width) != n_col_comps:
                raise ValueError(
                    f"Must have has many col comps as width values: {n_col_comps} != {len(width)}"
                )
            cols = [
                dbc.Col(comp, width=width_) for comp, width_ in zip(col_comps, width)
            ]
            row = dbc.Row(
                cols,
                align=self.align,
                justify=self.justify,
            )
            rows.append(row)

        if self.as_stack:
            return [dbc.Stack(rows, gap=self.row_gap)]

        return rows


class StackLayout(GridLayout):
    """Row with multiple columns or multiple rows with one column.

    Parameters
    ----------
    row_gap : int
        Gap between rows. Applies if ``as_col`` is ``False``.
    reverse : bool
        Whether to reverse components.
    """

    def __init__(
        self,
        width=None,
        align="center",
        justify="center",
        sorter=None,
        as_col=True,
        row_gap=5,
        reverse=False,
    ):
        if as_col:
            _row_to_col = lambda x: [x]
            sorter = compose_all(_row_to_col, sorter)

            width = [width]

        if reverse:
            _reversed = lambda x: list(reversed(x))
            sorter = compose_all(sorter, _reversed)

        super().__init__(
            sorter=sorter,
            width=width,
            align=align,
            justify=justify,
            as_stack=not as_col,
            row_gap=row_gap,
        )


class MriExplorerLayout(GridLayout):
    def __init__(self, as_col=True, graph_first=True):
        if as_col:
            width = [(6, 6)]
            sorter = lambda x: [
                [x[0][0], StackLayout(as_col=False, row_gap=5)([x[0][1], x[1][0]])]
            ]
        else:
            width = [None, (8, 4)]
            sorter = lambda x: [x[0][0], [x[0][1], x[1][0]]]

        if not graph_first:
            if as_col:
                _swap = lambda x: [[x[0][1], x[0][0]]]

            else:
                width = list(reversed(width))
                _swap = lambda x: [x[1], x[0]]

            sorter = compose_all(_swap, sorter)

        super().__init__(width=width, sorter=sorter)


class SwitchableMriViewLayout(StackLayout):
    def __init__(self, as_col=False, graph_first=True):
        if as_col:
            sorter = lambda x: [
                x[0],
                StackLayout(as_col=False, width=("auto", None))(x[1:]),
            ]
            width = [6, 6]
        else:
            sorter = lambda x: x
            width = [None, "auto", None]

        if not graph_first:
            perm = [1, 0] if as_col else [1, 2, 0]

            width = [width[index] for index in perm]
            _swap = lambda x: [x[index] for index in perm]

            sorter = compose_all(_swap, sorter)

        super().__init__(as_col=as_col, width=width, sorter=sorter)
