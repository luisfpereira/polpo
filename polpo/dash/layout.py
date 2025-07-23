import abc

import dash_bootstrap_components as dbc
from dash import html

from .style import STYLE as S

# TODO: add e.g. prefixed/suffixed layout?

# TODO: do some homogenization?
# TODO: generalize but keep syntax sugar

# TODO: Column(s) -> Col(s)


# TODO: take use of components layout to simplify this


class Layout(abc.ABC):
    def __init__(self, sorter=None):
        if sorter is None:
            sorter = lambda x: x
        self.sorter = sorter

    @abc.abstractmethod
    def __call__(self, comps):
        pass


class DummyLayout(Layout):
    def __call__(self, comps):
        return comps


class NestedLayout(Layout):
    def __init__(self, layouts, sorter=None):
        super().__init__(sorter=sorter)
        self.layouts = layouts

    def __call__(self, comps):
        comps = self.sorter(comps)
        for layout in self.layouts:
            comps = layout(comps)

        return comps


class TwoColumnLayout(Layout):
    def __call__(self, comps):
        right, left = self.sorter(comps)

        # TODO: revisit and delete comment
        # right_comp = dbc.Stack(
        #     right.to_dash(),
        #     gap=3,
        # )

        return [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            left,
                            style={"paddingTop": "0px"},
                        ),
                        sm=6,
                        width=900,
                    ),
                    dbc.Col(sm=3, width=100),
                    dbc.Col(right, sm=3, width=500),
                ],
                align="center",
                style={
                    "marginLeft": S.margin_side,
                    "marginRight": S.margin_side,
                    "marginTop": "50px",
                },
            ),
        ]


class SwappedTwoColumnLayout(Layout):
    def __call__(self, comps):
        left, right = self.sorter(comps)

        # TODO: revisit and delete comment
        # left_comp = dbc.Stack(
        #     left.to_dash(),
        #     gap=3,
        # )

        return [
            dbc.Row(
                [
                    dbc.Col(left, sm=3, width=500),
                    dbc.Col(sm=3, width=100),
                    dbc.Col(
                        html.Div(
                            right,
                            style={"paddingTop": "0px"},
                        ),
                        sm=6,
                        width=900,
                    ),
                ],
                align="center",
                style={
                    "marginLeft": S.margin_side,
                    "marginRight": S.margin_side,
                    "marginTop": "50px",
                },
            ),
        ]


class TwoRowLayout(Layout):
    def __call__(self, comps):
        top, bottom = self.sorter(comps)

        # TODO: revisit and delete comment
        # top_comp = dbc.Stack(
        #     top.to_dash(),
        #     gap=3,
        # )

        row_style = {
            "marginLeft": S.margin_side,
            "marginRight": S.margin_side,
            "marginTop": "50px",
        }
        return [
            dbc.Row(
                [dbc.Col(top, sm=3, width=500)],
                align="center",
                style=row_style,
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            bottom,
                            style={"paddingTop": "0px"},
                        ),
                        sm=6,
                        width=900,
                    )
                ],
                align="center",
                style=row_style,
            ),
        ]


class SwappedTwoRowLayout(Layout):
    def __call__(self, comps):
        bottom, top = self.sorter(comps)

        # TODO: revisit and delete comment
        # bottom_comp = dbc.Stack(
        #     bottom.to_dash(),
        #     gap=3,
        # )

        row_style = {
            "marginLeft": S.margin_side,
            "marginRight": S.margin_side,
            "marginTop": "50px",
        }
        return [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            top,
                            style={"paddingTop": "0px"},
                        ),
                        sm=6,
                        width=900,
                    )
                ],
                align="center",
                style=row_style,
            ),
            dbc.Row(
                [dbc.Col(bottom, sm=3, width=500)],
                align="center",
                style=row_style,
            ),
        ]


class GraphInputTwoColumnLayout(Layout):
    def __call__(self, comps):
        inputs, graph = self.sorter(comps)

        return [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            graph,
                            style={
                                "paddingTop": "0px",
                                "width": "100%",  # full width of this col
                                "maxWidth": "100%",  # prevent overflow
                            },
                        ),
                        xs=12,
                        sm=12,
                        md=6,  # full width on small screens, half on medium+
                        style={"padding": "10px"},
                    ),
                    dbc.Col(
                        html.Div(inputs),
                        xs=12,
                        sm=12,
                        md=6,  # full width on small screens, half on medium+
                        style={"padding": "10px"},
                    ),
                ],
                align="start",
                style={
                    "margin": "0 auto",
                    "width": "100%",
                    "maxWidth": "1200px",  # max total width of row
                    "flexWrap": "wrap",  # important for responsive stacking
                },
            )
        ]


class MultiRowLayout(Layout):
    # TODO: rename and add one for different top
    def __call__(self, comps):
        # NB: top is treated differently
        top, other = self.sorter(comps)

        # TODO: revisit and delete comment
        # top_comp = dbc.Stack(
        #     top.to_dash(),
        #     gap=3,
        # )

        top_row_style = {
            "marginLeft": S.margin_side,
            "marginRight": S.margin_side,
            "marginTop": "50px",
        }
        row_style = {
            "marginLeft": S.margin_side,
            "marginRight": S.margin_side,
        }
        return [
            dbc.Row(
                [dbc.Col(top, sm=3, width=500)],
                align="center",
                style=top_row_style,
            ),
            dbc.Stack(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    other_comp,
                                    style={"paddingTop": "0px"},
                                ),
                                sm=6,
                                width=900,
                                align="center",
                            )
                        ],
                        style=row_style,
                    )
                    for other_comp in other
                ],
                gap=0,
            ),
        ]


class OneTwoColLayout(Layout):
    # TODO: easy to generalize
    def __init__(
        self,
        sm=(14, 7, 4),
        width=(700, 700),
        top_margin="10px",
        sep_margin="10px",
        sorter=None,
    ):
        # width only applies for second row
        super().__init__(sorter=sorter)
        self.sm = sm
        self.width = width
        self.top_margin = top_margin
        self.sep_margin = sep_margin

    def __call__(self, comps):
        comps = self.sorter(comps)

        first = dbc.Row(
            [
                dbc.Col(
                    comps[0],
                    sm=self.sm[0],
                    width=sum(self.width),
                ),
            ],
            align="center",
            style={
                "marginLeft": S.margin_side,
                "marginRight": S.margin_side,
                "marginTop": self.top_margin,
            },
        )

        second_cols = [
            dbc.Col(comp, sm=sm, width=width)
            for comp, sm, width in zip(comps[1:], self.sm[1:], self.width)
        ]
        second = dbc.Row(
            second_cols,
            align="center",
            style={
                "marginLeft": S.margin_side,
                "marginRight": S.margin_side,
                "marginTop": self.sep_margin,
            },
        )
        return [first, second]


class OneColMultiRowLayout(Layout):
    # TODO: homogenize with multirow layout?
    def __init__(self, sep_margin="10px", sorter=None):
        super().__init__(sorter=sorter)
        self.sep_margin = sep_margin

    def __call__(self, comps):
        comps = self.sorter(comps)

        return [
            dbc.Row(
                [dbc.Col(comp, sm=14)],
                align="center",
                style={
                    "marginLeft": S.margin_side,
                    "marginRight": S.margin_side,
                    "marginTop": self.sep_margin,
                },
            )
            for comp in comps
        ]


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


class MultiColLayout(Layout):
    def __init__(
        self,
        sm=None,
        width=None,
        align="center",
        sorter=None,
        col_style=None,
        row_style=None,
    ):
        super().__init__(sorter=sorter)
        if col_style is None:
            col_style = {}

        if row_style is None:
            row_style = {}

        self.sm = sm
        self.align = align
        self.width = width

        self.col_style = {"padding": "20px"}.update(col_style)
        self.row_style = {
            "marginLeft": "10px",
            "marginRight": "10px",
            "marginTop": "50px",
        }.update(row_style)

    def __call__(self, comps):
        comps = self.sorter(comps)

        sm = self.sm
        if isinstance(sm, int) or sm is None:
            sm = [sm] * len(comps)

        width = self.width
        if isinstance(width, int) or width is None:
            width = [width] * len(comps)

        return [
            dbc.Row(
                [
                    dbc.Col(
                        comp,
                        sm=sm_,
                        width=width_,
                        style=self.col_style,
                    )
                    for comp, sm_, width_ in zip(comps, sm, width)
                ],
                align=self.align,
                style=self.row_style,
            )
        ]
