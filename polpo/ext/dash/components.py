"""Components."""

import abc

import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, get_asset_url, html

from polpo.models import (
    MriSlicesLookup,
    PdDfLookup,
)
from polpo.plot.mesh import MeshPlotter
from polpo.plot.mri import SlicePlotter
from polpo.utils import unnest_list

from .callbacks import (
    create_button_toggler_for_view_model_update,
    create_view_model_update,
)
from .layout import GraphInputTwoColumnLayout, MultiRowLayout, TwoColumnLayout
from .style import STYLE as S


class Component(abc.ABC):
    def __init__(self, id_prefix=""):
        self.id_prefix = id_prefix

    @abc.abstractmethod
    def to_dash(self):
        # NB: returns list[dash.Component]
        pass

    def as_output(self, component_property, allow_duplicate=False):
        return [Output(self.id, component_property, allow_duplicate=allow_duplicate)]

    def prefix(self, name):
        return f"{self.id_prefix}{name}"


class AdaptedDashComponent(Component):
    def __init__(self, comp):
        super().__init__()
        self.comp = comp

    @property
    def id(self):
        return self.comp.id

    def to_dash(self):
        return [self.comp]


class VarDefComponent(Component, abc.ABC):
    def __init__(self, var_def, id_prefix="", id_suffix=""):
        super().__init__(id_prefix)
        self.var_def = var_def
        self.id_suffix = id_suffix

    @property
    def id(self):
        return f"{self.id_prefix}{self.var_def.id}{self.id_suffix}"


class IdComponent(Component, abc.ABC):
    # TODO: improve these abstractions
    def __init__(self, id_, id_prefix="", id_suffix=""):
        super().__init__(id_prefix)
        self.id_ = id_
        self.id_suffix = id_suffix

    @property
    def id(self):
        return f"{self.id_prefix}{self.id_}{self.id_suffix}"


class DummyComponent(IdComponent):
    """Dummy component.

    Can be used in replacement of optional components.
    """

    def __init__(self):
        super().__init__(id_="dummy")

    def as_output(self, component_property=None, allow_duplicate=False):
        return []

    def as_input(self):
        return []

    def to_dash(self):
        return []


class BaseComponentGroup(Component, abc.ABC):
    def __init__(self, components, id_prefix=""):
        self.components = components
        super().__init__(id_prefix)

    def __getitem__(self, index):
        return self.components[index]

    def __len__(self):
        return len(self.components)

    @property
    def id_prefix(self):
        return self._id_prefix

    @id_prefix.setter
    def id_prefix(self, value):
        self._id_prefix = value
        if value:
            for component in self.components:
                component.id_prefix = value


class ComponentGroup(BaseComponentGroup):
    def __init__(self, components, id_prefix="", title=None, ordering=None):
        # ordering applies only to list[VarDefComponent]
        # ordering not only orders components, but also selects them
        # i.e. if they're not in the ordering, then will be dismissed
        # this is very important to avoid bugs during configuration
        if ordering is not None:
            components_ = []
            for id_ in ordering:
                for component in components:
                    if component.var_def.id == id_:
                        break
                else:
                    raise ValueError(f"{id_} not found in components.")
                components_.append(component)

            components = components_

        super().__init__(components, id_prefix)
        self.title = title

    def to_dash(self, data=None):
        if data is not None:
            return unnest_list(
                [component.to_dash(value) for component, value in zip(self, data)]
            )

        # TODO: should this be part of the components?
        # e.g. split between update and not updatable components
        title_label = (
            [
                dbc.Label(
                    f"{self.title}:",
                    style={
                        "font-size": S.text_fontsize,
                        "fontFamily": S.text_fontfamily,
                    },
                ),
            ]
            if self.title
            else []
        )

        return title_label + unnest_list([component.to_dash() for component in self])

    def as_output(self, component_property=None, allow_duplicate=False):
        return unnest_list(
            component.as_output(
                component_property=component_property, allow_duplicate=allow_duplicate
            )
            for component in self
        )

    def as_empty_output(self):
        return unnest_list(component.as_empty_output() for component in self)

    def as_input(self):
        return unnest_list(component.as_input() for component in self)


class RadioButton(Component):
    """Radio button group.

    Parameters
    ----------
    id_ : str
        The unique ID for the radio button group.
    options : list of tuple
        A list of (value, label) tuples for the options.
    default_value : str
        The default selected value.
    inline : bool
        Whether to display options inline (horizontally).
    """

    def __init__(self, id_, options, default_value=None, inline=True):
        super().__init__(id_=id_)
        self.options = options
        self.default_value = default_value or options[0][0]
        self.inline = inline

    def to_dash(self):
        """Convert the component into a Dash UI element."""
        return [
            dbc.FormGroup(
                [
                    dcc.RadioItems(
                        id=self.id,
                        options=[
                            {"label": label, "value": value}
                            for value, label in self.options
                        ],
                        value=self.default_value,
                        inline=self.inline,
                    )
                ]
            )
        ]

    def as_input(self):
        return [Input(self.id, "value")]

    def as_output(self, component_property="value", allow_duplicate=False):
        return [Output(self.id, component_property, allow_duplicate=allow_duplicate)]


class Checkbox(Component):
    """Checkbox.

    Parameters
    ----------
    id_ : str
        The unique ID for the checkbox.
    label : str
        The label displayed next to the checkbox.
    default_checked : bool
        Whether the checkbox should be checked by default.
    """

    def __init__(self, id_, label="Show", default_checked=True):
        super().__init__(id_=id_)
        self.label = label
        self.default_checked = default_checked

    def to_dash(self):
        """Convert the component into a Dash UI element."""
        return [
            dbc.FormGroup(
                [
                    dcc.Checklist(
                        id=self.id_,
                        options=[{"label": self.label, "value": "checked"}],
                        value=["checked"] if self.default_checked else [],
                        inline=True,
                    )
                ]
            )
        ]


class Slider(VarDefComponent):
    """Slider."""

    def __init__(self, var_def, step=1, id_prefix="", label_style=None):
        super().__init__(var_def=var_def, id_prefix=id_prefix, id_suffix="-slider")
        # TODO: think more about this design

        self.step = step

        # TODO: can default be set for the general app instead?
        default_label_style = {
            "fontSize": S.text_fontsize,
            "fontFamily": S.text_fontfamily,
        }
        self.label_style = (label_style or {}).update(default_label_style)

    def __repr__(self):
        return f"Slider({self.id})"

    def to_dash(self):
        # TODO: allow to config from config file, e.g. label_style
        label = dbc.Label(
            self.var_def.label,
            style=self.label_style,
        )

        # ensure default value is on the slider
        min_value, max_value = self.var_def.min_value, self.var_def.max_value
        step = self.step
        value = min(max_value, self.var_def.default_value)
        value = max(min_value, value)
        n_steps = round((value - min_value) / step)
        value = min_value + step * n_steps

        slider = dcc.Slider(
            id=self.id,
            min=min_value,
            max=max_value,
            step=step,
            value=value,
            marks={
                self.var_def.min_value: {"label": "min"},
                self.var_def.max_value: {"label": "max"},
            },
            tooltip={
                "placement": "bottom",
                "always_visible": True,
                "style": {"fontSize": "25px", "fontFamily": S.text_fontfamily},
            },
        )

        return [label, slider]

    def as_input(self):
        return [Input(self.id, "drag_value")]


class DepVar(VarDefComponent):
    def __repr__(self):
        return f"DepVar({self.var_def.id})"

    def to_dash(self, value=None):
        if value:
            return [f"{self.var_def.label}: {value}"]

        return [
            html.Div(
                id=self.id,
                style={
                    "font-size": S.text_fontsize,
                    "fontFamily": S.text_fontfamily,
                },
            )
        ]

    def as_output(self, component_property=None, allow_duplicate=False):
        component_property = component_property or "children"
        return [Output(self.id, component_property, allow_duplicate=allow_duplicate)]

    def as_empty_output(self):
        return [""]


class Graph(IdComponent):
    def __init__(self, id_, plotter=None, id_prefix="", id_suffix=""):
        # TODO: add reasonable default plotter
        super().__init__(id_, id_prefix, id_suffix)
        self.plotter = plotter
        self.graph_ = None

    def to_dash(self, data=None):
        if self.graph_ is not None:
            return [self.plotter.update(self.graph_.figure, data)]
        self.graph_ = dcc.Graph(
            id=self.id,
            config={"displayModeBar": False, "responsive": True},
            figure=self.plotter.plot(data),
            style={
                "aspectRatio": "1",
                "width": "25vw",
                "height": "auto",
            },
        )
        return [self.graph_]

    def as_output(self, component_property=None, allow_duplicate=False):
        component_property = component_property or "figure"
        return [Output(self.id, "figure", allow_duplicate=allow_duplicate)]

    def as_empty_output(self):
        return [self.plotter.plot()]


class Image(IdComponent):
    def __init__(self, id_, id_prefix="", id_suffix="", style=None):
        super().__init__(id_, id_prefix, id_suffix)

        if style is None:
            style = {"width": "100%"}

        self._image = html.Img(id=self.id_, src="", style=style)

    def to_dash(self, data=None):
        if data is not None:
            return [data]

        return [self._image]

    def as_output(self, component_property="src", allow_duplicate=False):
        return [Output(self.id, component_property, allow_duplicate=allow_duplicate)]

    def as_empty_output(self):
        return [""]


class GraphRow(ComponentGroup):
    def __init__(self, n_graphs=3, graphs=None, id_prefix=""):
        # NB: `n_graphs`` is ignored if `graphs` is not None

        if graphs is None:
            graphs = [
                Graph(id_="plot", id_suffix=f"-{index}") for index in range(n_graphs)
            ]

        super().__init__(components=graphs, id_prefix=id_prefix)

    def to_dash(self, data=None):
        if data is not None:
            return super().to_dash(data)

        return [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            graph.to_dash(),
                            style={"paddingTop": "0px"},
                        ),
                        sm=4,
                    )
                    for graph in self
                ],
                align="center",
                style={
                    "marginLeft": "10px",
                    "marginRight": "10px",
                    "marginTop": "50px",
                },
            )
        ]


class MriSliders(ComponentGroup):
    def __init__(self, components, trims=((20, 40), 50, 70), id_prefix="", title=None):
        super().__init__(components, id_prefix=id_prefix, title=title)
        self.trims = [(trim, trim) if isinstance(trim, int) else trim for trim in trims]

    def update_lims(self, mri_data):
        self[0].var_def.default_value = self[0].var_def.default_value or 1
        self[0].var_def.max_value = min(self[0].var_def.max_value, len(mri_data))
        for slider, trim in zip(self[1:], self.trims):
            var_def = slider.var_def
            min_value = var_def.min_value or trim[0]
            max_value = var_def.max_value or (mri_data[0].shape[0] - 1 - trim[1])
            var_def.default_value = (
                var_def.default_value or (max_value - min_value) // 2 + min_value
            )
            var_def.min_value = min_value
            var_def.max_value = max_value


class MriGraphRow(GraphRow):
    # NB: just syntax sugar

    def __init__(self, index_ordering=(0, 1, 2)):
        titles = ("Side View", "Front View", "Top View")
        x_labels = ("Y", "X", "X")
        y_labels = ("Z", "Z", "Y")

        titles = [titles[index] for index in index_ordering]
        x_labels = [x_labels[index] for index in index_ordering]
        y_labels = [y_labels[index] for index in index_ordering]

        graphs = [
            Graph(
                id_="plot",
                id_suffix=f"-{index}",
                plotter=SlicePlotter(title=title, x_label=x_label, y_label=y_label),
            )
            for index, (title, x_label, y_label) in enumerate(
                zip(titles, x_labels, y_labels)
            )
        ]
        super().__init__(id_prefix="nii-", graphs=graphs)


class MriExplorer(BaseComponentGroup):
    # data
    # plots
    # sliders
    # session info card

    # also instructions?

    # TODO: check if multiple callbacks can be defined
    def __init__(
        self,
        mri_data,
        hormones_df,
        sliders,
        session_info,
        graph_row=None,
        id_prefix="",
    ):
        if graph_row is None:
            graph_row = MriGraphRow(index_ordering=list(range(len(sliders) - 1)))

        # TODO: used to train the model and to update the controller
        self.mri_data = mri_data
        self.hormones_df = hormones_df

        # NB: an input view
        self.sliders = sliders
        # NB: an output view of the brain data
        self.graph_row = graph_row
        # NB: a model of the brain data
        self.mri_model = MriSlicesLookup(self.mri_data)

        # NB: a view of the hormones data
        self.session_info = session_info
        # NB: a model of the hormones data
        self.session_info_model = PdDfLookup(
            df=hormones_df,
            output_keys=[elem.var_def.id for elem in session_info],
            tar=1,
        )

        super().__init__([sliders, graph_row, session_info], id_prefix)

    def _create_callbacks(self):
        create_view_model_update(self.sliders, self.graph_row, self.mri_model)
        create_view_model_update(
            self.sliders[0], self.session_info, self.session_info_model
        )

    def to_dash(self):
        if hasattr(self.sliders, "update_lims"):
            self.sliders.update_lims(self.mri_data)

        plots_card = self.graph_row.to_dash()
        plots = dbc.Row(
            [
                dbc.Col(plots_card, sm=14),
            ],
            align="center",
            style={
                "marginLeft": S.margin_side,
                "marginRight": S.margin_side,
                "marginTop": "50px",
            },
        )

        sliders_card = dbc.Card(
            [
                dbc.Stack(
                    self.sliders.to_dash(),
                    gap=3,
                )
            ],
            body=True,
        )
        sliders_column = [
            dbc.Row(sliders_card),
        ]

        session_info = self.session_info.to_dash()
        sess_info_card = dbc.Card(
            [
                dbc.Stack(
                    session_info,
                    gap=0,
                )
            ],
            body=True,
        )

        sliders_and_session = dbc.Row(
            [
                dbc.Col(sliders_column, sm=7, width=700),
                dbc.Col(sess_info_card, sm=4, width=700),
            ],
            align="center",
            style={
                "marginLeft": S.margin_side,
                "marginRight": S.margin_side,
                "marginTop": "50px",
            },
        )

        self._create_callbacks()

        return [plots, sliders_and_session]


class ModelBasedExplorer(BaseComponentGroup):
    def __init__(
        self, model, inputs, output, id_prefix="", postproc_pred=None, layout=None
    ):
        if layout is None:
            layout = TwoColumnLayout()

        self.model = model
        self.inputs = inputs
        self.output = output
        self.postproc_pred = postproc_pred
        self.layout = layout

        super().__init__([output, inputs], id_prefix=id_prefix)

    def to_dash(self):
        out = self.layout.to_dash([self.inputs, self.output])

        create_view_model_update(
            output_view=self.output,
            input_view=self.inputs,
            model=self.model,
            postproc_pred=self.postproc_pred,
        )

        return out


class ImageExplorer(ModelBasedExplorer):
    def __init__(self, model, inputs, image=None, id_prefix="", layout=None):
        if image is None:
            image = Image(id_="image-expl", id_prefix=id_prefix)

        super().__init__(model, inputs, image, id_prefix=id_prefix, layout=layout)


class MeshExplorer(ModelBasedExplorer):
    def __init__(
        self, model, inputs, graph=None, id_prefix="", postproc_pred=None, layout=None
    ):
        if graph is None:
            graph = Graph(id_="mesh-plot", plotter=MeshPlotter(), id_prefix=id_prefix)

        self.model = model
        self.graph = graph
        self.inputs = inputs
        self.postproc_pred = postproc_pred

        super().__init__(
            model,
            inputs,
            graph,
            id_prefix=id_prefix,
            postproc_pred=postproc_pred,
            layout=layout,
        )


class SharedInputModelsBasedExplorer(BaseComponentGroup):
    def __init__(
        self, models, inputs, outputs, id_prefix="", postproc_pred=None, layout=None
    ):
        if layout is None:
            layout = MultiRowLayout()

        self.models = models
        self.inputs = inputs
        self.outputs = outputs
        self.postproc_pred = postproc_pred
        self.layout = layout

        super().__init__([outputs, inputs], id_prefix=id_prefix)

    def to_dash(self):
        out = self.layout.to_dash([self.inputs, self.outputs])

        for output_, model in zip(self.outputs, self.models):
            create_view_model_update(
                output_view=output_,
                input_view=self.inputs,
                model=model,
                postproc_pred=self.postproc_pred,
            )

        return out


class MultiModelsMeshExplorer(BaseComponentGroup):
    def __init__(
        self,
        models,
        inputs,
        graph=None,
        id_prefix="",
        button_label="Switch model",
        checkbox_labels=None,
        postproc_pred=None,
        layout=None,
    ):
        # ignores button if only one model

        # TODO: add verifications?
        if graph is None:
            graph = Graph(id_="mesh-plot", plotter=MeshPlotter(), id_prefix=id_prefix)

        if layout is None:
            layout = GraphInputTwoColumnLayout()

        self.graph = graph
        self.models = models
        self.inputs = inputs
        self.button_label = button_label
        # NB: controls visibility of plots
        self.checkbox_labels = checkbox_labels
        self.postproc_pred = postproc_pred
        self.layout = layout

        super().__init__([self.graph].extend(self.inputs), id_prefix=id_prefix)

    def to_dash(self):
        inputs_cards = ComponentGroup(
            [
                HideableComponent(
                    id_=f"{index}_slider_container",
                    dash_component=dbc.Card(
                        dbc.Stack(
                            component.to_dash(),
                            gap=3,
                        ),
                        body=True,
                    ),
                    id_prefix=self.id_prefix,
                )
                for index, component in enumerate(self.inputs)
            ]
        )

        toggle_id = self.prefix("switch-model-button") if len(self.models) > 1 else None
        checkbox_id = self.prefix("show-model-checkbox")

        if self.checkbox_labels:
            # TODO: allow control of default visibility?
            n_graphs = self.graph.plotter.n_graphs

            checkbox_labels = [
                label
                if len(label) == 3
                else (label[0], label[1], label[0] < n_graphs - 1)
                for label in self.checkbox_labels
            ]
            checklist = Checklist(
                id_=checkbox_id,
                checkbox_labels=checkbox_labels,
                # NB: assumes graph has plotter with n_graphs
                n_options=n_graphs,
            )
        else:
            checklist = DummyComponent()

        button = (
            AdaptedDashComponent(
                html.Button(
                    self.button_label,
                    id=toggle_id,
                    n_clicks=0,
                )
            )
            if toggle_id
            else DummyComponent()
        )

        inputs = ComponentGroup([button, checklist, inputs_cards])

        out = self.layout.to_dash([inputs, self.graph])

        create_button_toggler_for_view_model_update(
            output_view=self.graph,
            input_views=self.inputs,
            models=self.models,
            toggle_id=toggle_id,
            checklist=checklist,
            hideable_components=inputs_cards,
            postproc_pred=self.postproc_pred,
        )

        return out


class HideableComponent(IdComponent):
    def __init__(self, id_, dash_component, id_prefix="", id_suffix=""):
        super().__init__(id_, id_prefix, id_suffix)
        self.dash_component = dash_component

    def to_dash(self):
        return [
            html.Div(
                children=[self.dash_component],
                id=self.id,
            )
        ]


class Checklist(IdComponent):
    def __init__(
        self,
        id_,
        checkbox_labels,
        n_options=None,
        inline=False,
        id_prefix="",
        id_suffix="",
    ):
        super().__init__(id_, id_prefix, id_suffix)

        self.inline = inline

        self.n_options = n_options or len(checkbox_labels)

        false_indices = []
        existing_indices = []
        self.options = []
        for option in checkbox_labels:
            option_ = {
                "label": option[1],
                "value": option[0] if option[0] >= 0 else self.n_options + option[0],
            }
            self.options.append(option_)
            existing_indices.append(option_["value"])
            visible = option[2] if len(option) > 2 else False
            if not visible:
                false_indices.append(option_["value"])

        self._default_bool_vis = [
            False if index in false_indices else True for index in range(n_options)
        ]
        self._default_bool = [
            False if index in existing_indices else True for index in range(n_options)
        ]

    def to_dash(self):
        # NB: defaults to uncheck if not specified
        return [
            dcc.Checklist(
                id=self.id,
                options=self.options,
                value=self.as_value(self._default_bool_vis),
                inline=self.inline,
            )
        ]

    def as_bool(self, value):
        # NB: updates read from callbacks
        bool_ls = self._default_bool.copy()

        for value_ in value:
            bool_ls[value_] = True

        return bool_ls

    def as_value(self, value):
        return [index for index, value_ in enumerate(value) if value_]

    def as_input(self):
        return [Input(self.id, "value")]


class SidebarHeader(Component):
    """Sidebar header.

    A row of a sidebar.
    Controls text and linking to page.

    Parameters
    ----------
    href : str
        Navigation link for page.
    text : str
        Text on the sidebar.
    image_url : str
        Image to appear on the sidebar.
    image_width : int
        Image width in pixel.
    """

    def __init__(self, href, text, image_url, image_width=30):
        super().__init__()
        self.href = href
        self.text = text
        self.image_url = image_url
        self.image_width = image_width

    def to_dash(self):
        return [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(
                            src=get_asset_url(self.image_url),
                            style={"width": f"{self.image_width}px", "height": "auto"},
                        ),
                        width=2,
                    ),
                    dbc.Col(
                        dbc.NavLink(self.text, href=self.href, active="exact"),
                        width=10,
                    ),
                ],
                align="center",
            )
        ]


class FunctionComponent(Component):
    """Component that wraps a function.

    Allows to wrap functions that create dash components.
    For compatibility with dash component.

    Parameters
    ----------
    func : callable
        Function being wrapped.
    """

    def __init__(self, func, **kwargs):
        super().__init__()
        self.func = func
        self.kwargs = kwargs

    def to_dash(self):
        return self.func(**self.kwargs)


class SidebarElem(Component):
    """Sidebar element.

    Composed of a header and a page.

    Parameters
    ----------
    tab_header : SidebarHeader
        Row of a sidebar.
    page : Component
        Page to which sidebar links.
    active : bool
        Whether the element is active.
    """

    def __init__(self, tab_header, page, active=True):
        super().__init__()
        self.active = active
        self.tab_header = tab_header
        self.page = page

    def to_dash(self, page_register):
        compns = self.tab_header.to_dash() + self.page.to_dash()

        # NB: assume page is only one element
        page_register.add_page(self.tab_header.href, compns[-1])

        return compns
