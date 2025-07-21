import abc

import dash_bootstrap_components as dbc
from dash import html

from .style import STYLE as S

# TODO: add e.g. prefixed/suffixed layout?

# TODO: do some homogenization?
# TODO: generalize but keep syntax sugar

# TODO: Column(s) -> Col(s)

# TODO: homogenize use of to_dash()


class Layout(abc.ABC):
    @abc.abstractmethod
    def to_dash(self, comps):
        pass


class DummyLayout(Layout):
    def to_dash(self, comps):
        return comps


class TwoColumnLayout(Layout):
    def to_dash(self, comps):
        right, left = comps

        left_comp = left.to_dash()
        # TODO: remove stack? add simple way of doing it in component?
        right_comp = dbc.Stack(
            right.to_dash(),
            gap=3,
        )

        return [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            left_comp,
                            style={"paddingTop": "0px"},
                        ),
                        sm=6,
                        width=900,
                    ),
                    dbc.Col(sm=3, width=100),
                    dbc.Col(right_comp, sm=3, width=500),
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
    def to_dash(self, comps):
        left, right = comps

        right_comp = right.to_dash()
        left_comp = dbc.Stack(
            left.to_dash(),
            gap=3,
        )

        return [
            dbc.Row(
                [
                    dbc.Col(left_comp, sm=3, width=500),
                    dbc.Col(sm=3, width=100),
                    dbc.Col(
                        html.Div(
                            right_comp,
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
    def to_dash(self, comps):
        top, bottom = comps

        bottom_comp = bottom.to_dash()
        top_comp = dbc.Stack(
            top.to_dash(),
            gap=3,
        )

        row_style = {
            "marginLeft": S.margin_side,
            "marginRight": S.margin_side,
            "marginTop": "50px",
        }
        return [
            dbc.Row(
                [dbc.Col(top_comp, sm=3, width=500)],
                align="center",
                style=row_style,
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            bottom_comp,
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
    def to_dash(self, comps):
        bottom, top = comps

        top_comp = top.to_dash()
        bottom_comp = dbc.Stack(
            bottom.to_dash(),
            gap=3,
        )

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
                            top_comp,
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
                [dbc.Col(bottom_comp, sm=3, width=500)],
                align="center",
                style=row_style,
            ),
        ]


class GraphInputTwoColumnLayout(Layout):
    def to_dash(self, comps):
        inputs, graph = comps

        graph_comp = graph.to_dash()
        inputs_comp = inputs.to_dash()

        return [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            graph_comp,
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
                        html.Div(inputs_comp),
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
    def to_dash(self, comps):
        # NB: top is treated differently
        top, other = comps

        top_comp = dbc.Stack(
            top.to_dash(),
            gap=3,
        )
        other_comps = [other_.to_dash() for other_ in other]

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
                [dbc.Col(top_comp, sm=3, width=500)],
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
                    for other_comp in other_comps
                ],
                gap=0,
            ),
        ]


class OneTwoColumnsLayout(Layout):
    # TODO: easy to generalize
    def __init__(
        self, sm=(14, 7, 4), width=(700, 700), top_margin="10px", sep_margin="10px"
    ):
        # width only applies for second row
        super().__init__()
        self.sm = sm
        self.width = width
        self.top_margin = top_margin
        self.sep_margin = sep_margin

    def to_dash(self, comps):
        first = dbc.Row(
            [
                dbc.Col(comps[0], sm=14),
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


class OneOneColumnsLayout(Layout):
    # TODO: homogenize with multirow layout?
    def __init__(self, sep_margin="10px"):
        super().__init__()
        self.sep_margin = sep_margin

    def to_dash(self, comps):
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
    def __init__(self, gap=3):
        super().__init__()
        self.gap = gap

    def to_dash(self, comps):
        return [
            dbc.Card(
                [dbc.Stack(comps, gap=self.gap)],
                body=True,
            )
        ]


class MultiColumnLayout(Layout):
    def __init__(self, sm=4, align="center"):
        super().__init__()
        self.sm = sm
        self.align = align

    def to_dash(self, comps):
        return [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            comp,
                            style={"paddingTop": "0px"},
                        ),
                        sm=self.sm,
                    )
                    for comp in comps
                ],
                align=self.align,
                style={
                    "marginLeft": "10px",
                    "marginRight": "10px",
                    "marginTop": "50px",
                },
            )
        ]
