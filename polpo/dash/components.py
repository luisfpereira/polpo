"""Components."""

import abc

import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, get_asset_url, html

from polpo.dash.callbacks import (
    ButtonTogglerForViewModelUpdateFactory,
    ViewModelUpdateFactory,
)
from polpo.dash.layout import (
    DummyLayout,
    GridLayout,
    StackInCard,
    StackLayout,
)
from polpo.dash.style import STYLE as S
from polpo.dash.variables import VarDef
from polpo.models import (
    MriSlicesLookup,
    PdDfLookup,
    SwitchableMriSlicesLookup,
)
from polpo.plot.mesh import MeshPlotter
from polpo.plot.mri import SlicePlotter
from polpo.utils import compose_all, unnest_list

# TODO: rename callbacks to callbacks_factory


class Component(abc.ABC):
    def __init__(self, id_prefix=""):
        self.id_prefix = id_prefix

        self._dash_component = None

    @abc.abstractmethod
    def to_dash(self):
        # NB: returns list[dash.Component]
        pass

    def as_output(self, component_property, allow_duplicate=False):
        return [Output(self.id, component_property, allow_duplicate=allow_duplicate)]

    def prefix(self, name):
        # TODO: may want to remove this
        return f"{self.id_prefix}{name}"


class AdaptedDashComponent(Component):
    def __init__(self, comp):
        super().__init__()
        self._dash_component = comp

    @property
    def id(self):
        return self._dash_component.id

    def to_dash(self):
        return [self._dash_component]


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


class DummyComponent(Component):
    """Dummy component.

    Can be used in replacement of optional components.
    """

    def __init__(self):
        super().__init__()

    @property
    def id(self):
        return None

    def as_output(self, component_property=None, allow_duplicate=False):
        return []

    def as_input(self):
        return []

    def to_dash(self):
        return []


class BaseComponentGroup(Component):
    def __init__(
        self, components, id_prefix="", layout=None, callbacks=(), unnest=True
    ):
        if layout is None:
            layout = DummyLayout()

        self.components = components
        self.callbacks = callbacks
        self.layout = layout
        self.unnest = unnest
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

    def to_dash(self):
        if self._dash_component is not None:
            return self._dash_component

        dash_comps = [comp.to_dash() for comp in self.components]
        if self.unnest:
            dash_comps = unnest_list(dash_comps)

        self._dash_component = self.layout(dash_comps)

        for callback in self.callbacks:
            callback()

        return self._dash_component

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


class ComponentGroup(BaseComponentGroup):
    # TODO: rename and use this name for base?

    def __init__(
        self, components, id_prefix="", title=None, ordering=None, layout=None
    ):
        # ordering applies only to list[VarDefComponent]
        # ordering not only orders components, but also selects them
        # i.e. if they're not in the ordering, then they will be dismissed
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

        if layout is None:
            layout = DummyLayout()

        super().__init__(components, id_prefix)
        self.title = title
        self.layout = layout

    def to_dash(self, data=None):
        if data is not None:
            return unnest_list(
                [component.to_dash(value) for component, value in zip(self, data)]
            )

        # TODO: check if behavior is not broken due to cache
        if self._dash_component is not None:
            return self._dash_component

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

        self._dash_component = self.layout(
            title_label + unnest_list([component.to_dash() for component in self])
        )
        return self._dash_component


class RadioButton(IdComponent):
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

    def __init__(self, id_, options, default_value=None, inline=True, layout=None):
        super().__init__(id_=id_)
        self.options = options
        self.default_value = default_value or options[0][0]
        self.inline = inline

        if layout is None:
            layout = DummyLayout()

        self.layout = layout

    def to_dash(self):
        """Convert the component into a Dash UI element."""
        if self._dash_component is not None:
            return self._dash_component

        self._dash_component = self.layout(
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

        return self._dash_component

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
        if self._dash_component is not None:
            return self._dash_component

        self._dash_component = [
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
        return self._dash_component


class Slider(VarDefComponent):
    """Slider."""

    def __init__(self, var_def, step=1, id_prefix="", label_style=None, layout=None):
        # TODO: think more about this design
        # TODO: does var_def really makes sense? don't think so
        # TODO: make it has a composite component for easier control of layout?

        if layout is None:
            layout = DummyLayout()

        self.step = step

        # TODO: can default be set for the general app instead?
        default_label_style = {
            "fontSize": S.text_fontsize,
            "fontFamily": S.text_fontfamily,
        }
        self.label_style = (label_style or {}).update(default_label_style)
        self.layout = layout

        super().__init__(var_def=var_def, id_prefix=id_prefix, id_suffix="-slider")

    def __repr__(self):
        return f"Slider({self.id})"

    def to_dash(self):
        if self._dash_component is not None:
            return self._dash_component

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

        self._dash_component = self.layout([label, slider])

        return self._dash_component

    def as_input(self):
        return [Input(self.id, "drag_value")]


class DepVar(VarDefComponent):
    def __repr__(self):
        return f"DepVar({self.var_def.id})"

    def to_dash(self, value=None):
        if value:
            return [f"{self.var_def.label}: {value}"]

        if self._dash_component is not None:
            return self._dash_component

        self._dash_component = [
            html.Div(
                id=self.id,
                style={
                    "font-size": S.text_fontsize,
                    "fontFamily": S.text_fontfamily,
                },
            )
        ]
        return self._dash_component

    def as_output(self, component_property=None, allow_duplicate=False):
        component_property = component_property or "children"
        return [Output(self.id, component_property, allow_duplicate=allow_duplicate)]

    def as_empty_output(self):
        return [""]


class Graph(IdComponent):
    def __init__(self, id_, plotter=None, id_prefix="", id_suffix="", layout=None):
        # TODO: add reasonable default plotter or remove None
        if layout is None:
            layout = lambda comp: html.Div(
                comp,
                style={
                    "paddingTop": "0px",
                    "width": "100%",
                    "maxWidth": "100%",
                },
            )

        super().__init__(id_, id_prefix, id_suffix)
        self.plotter = plotter
        self.graph_ = None
        self.layout = layout

    def to_dash(self, data=None):
        if self.graph_ is not None:
            return [self.plotter.update(self.graph_.figure, data)]

        if self._dash_component is not None:
            return self._dash_component

        self.graph_ = dcc.Graph(
            id=self.id,
            config={"displayModeBar": False, "responsive": True},
            figure=self.plotter.plot(data),
            style={
                "aspectRatio": "1",
                "width": "100%",
            },
        )
        self._dash_component = self.layout([self.graph_])
        return self._dash_component

    def as_output(self, component_property=None, allow_duplicate=False):
        component_property = component_property or "figure"
        return [Output(self.id, "figure", allow_duplicate=allow_duplicate)]

    def as_empty_output(self):
        return [self.plotter.plot()]


class Image(IdComponent):
    def __init__(self, id_, id_prefix="", id_suffix="", style=None, layout=None):
        super().__init__(id_, id_prefix, id_suffix)

        if layout is None:
            layout = DummyLayout()

        style = {"width": "100%"}
        if style is None:
            style = style.update(style)

        self.layout = layout
        self._image = html.Img(id=self.id_, src="", style=style)

    def to_dash(self, data=None):
        if data is not None:
            return [data]

        return self.layout([self._image])

    def as_output(self, component_property="src", allow_duplicate=False):
        return [Output(self.id, component_property, allow_duplicate=allow_duplicate)]

    def as_empty_output(self):
        return [""]


class GraphStack(ComponentGroup):
    """A stack of graphs.

    Parameters
    ----------
    n_graphs : int
        Number of graphs. Ignored if ``graphs``.
    as_col : bool
        Whether to display graphs in columns. Ignored if ``layout``.
    """

    def __init__(self, n_graphs=3, graphs=None, id_prefix="", layout=None, as_col=True):
        if graphs is None:
            graphs = [
                Graph(id_="plot", id_suffix=f"-{index}") for index in range(n_graphs)
            ]

        if layout is None:
            layout = StackLayout(as_col=as_col)

        super().__init__(components=graphs, id_prefix=id_prefix)
        self.layout = layout

    def to_dash(self, data=None):
        if data is not None:
            return super().to_dash(data)

        # TODO: check if behavior is not broken due to cache
        if self._dash_component is not None:
            return self._dash_component

        self._dash_component = self.layout([graph.to_dash() for graph in self])
        return self._dash_component


class MriSliders(ComponentGroup):
    # TODO: delete?
    def __init__(
        self,
        components,
        trims=((20, 40), 50, 70),
        id_prefix="",
        title=None,
        layout=None,
    ):
        if layout is None:
            layout = StackInCard(gap=3)

        super().__init__(components, id_prefix=id_prefix, title=title, layout=layout)

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


class MriGraphStack(GraphStack):
    """A stack of MRI graphs.

    Parameters
    ----------
    as_col : bool
        Whether to display graphs in columns. Ignored if ``layout``.
    """

    def __init__(self, index_ordering=(0, 1, 2), layout=None, as_col=True):
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
        super().__init__(id_prefix="nii-", graphs=graphs, layout=layout, as_col=as_col)


class MriView(BaseComponentGroup):
    """View of MRI data.

    Displays MRI plots together with inputs for slice and (optionally) session.

    Parameters
    ----------
    stack_session : bool
        Whether to stack session input with slice input.
        If ``False``, assumes ``session_input`` component is created outside.
    as_col : bool
        Controls both global layout and graph_stack layout. Ignored if both passed.
    graph_first : bool
        Whether to display graph on top/left. Ignored if ``layout``.
    """

    def __init__(
        self,
        mri_data,
        session_input,
        slice_input=None,
        graph_stack=None,
        id_prefix="",
        stack_session=True,
        layout=None,
        as_col=False,
        graph_first=True,
    ):
        if slice_input is None:
            # TODO: update lims
            slice_input = ComponentGroup(
                [
                    Slider(
                        var_def=VarDef(
                            id_="mri_x",
                            name="X Coordinate (Changes Side View)",
                            min_value=20,
                            max_value=190,
                            default_value=(20 + 190) // 2,
                        ),
                        step=5,
                    ),
                    Slider(
                        var_def=VarDef(
                            id_="mri_y",
                            name="Y Coordinate (Changes Front View)",
                            min_value=25,
                            max_value=230,
                            default_value=(25 + 230) // 2,
                        ),
                        step=5,
                    ),
                    Slider(
                        var_def=VarDef(
                            id_="mri_z",
                            name="Z Coordinate (Changes Top View)",
                            min_value=72,
                            max_value=242,
                            default_value=(72 + 242) // 2,
                        ),
                        step=5,
                    ),
                ]
            )

        if graph_stack is None:
            graph_stack = MriGraphStack(
                index_ordering=list(range(len(slice_input.components))),
                as_col=not as_col,
            )

        mri_input = (
            ComponentGroup([session_input, slice_input], layout=StackInCard())
            if stack_session
            else InputGroup([session_input, slice_input])
        )

        mri_model = MriSlicesLookup(mri_data)
        graph_callback = ViewModelUpdateFactory(
            mri_input,
            graph_stack,
            mri_model,
        )

        components = (
            [graph_stack, mri_input] if stack_session else [graph_stack, slice_input]
        )

        if layout is None:
            sorter = None if graph_first else (lambda x: list(reversed(x)))
            layout = StackLayout(as_col=as_col, sorter=sorter)

        super().__init__(
            components,
            id_prefix=id_prefix,
            layout=layout,
            callbacks=[graph_callback],
        )


class SwitchableMriView(BaseComponentGroup):
    """View of MRI data.

    Displays MRI plot together with inputs for slice and (optionally) session.
    Differs from ``MriView`` as only one plot is displayed at a time
    (controlled by a radio button).

    Parameters
    ----------
    stack_session : bool
        Whether to stack session input with slice input.
        If ``False``, assumes ``session_input`` component is created outside.
    """

    def __init__(
        self,
        mri_data,
        session_input,
        slice_input=None,
        view_input=None,
        graph=None,
        id_prefix="",
        stack_session=True,
        layout=None,
        as_col=False,
        graph_first=True,
    ):
        if view_input is None:
            view_input = RadioButton(
                id_="mri-view-toggle",
                options=[(0, "Sagittal"), (1, "Coronal"), (2, "Axial")],
                layout=lambda comps: [
                    html.Div(
                        [
                            html.Span(
                                "MRI View",
                                style={"marginRight": "16px", "fontWeight": "bold"},
                            ),
                        ]
                        + comps,
                        style={
                            "display": "flex",
                        },
                    )
                ],
            )

        if slice_input is None:
            slice_input = Slider(
                VarDef(
                    id_="mri_coord",
                    name="MRI Slice",
                    min_value=20,
                    max_value=170,
                    default_value=(20 + 170) // 2,
                ),
                step=5,
                layout=DummyLayout()
                if stack_session
                else (lambda comps: dbc.Card(comps, body=True)),
            )

        if graph is None:
            graph = Graph(
                id_="plot", plotter=SlicePlotter(title=None, x_label=None, y_label=None)
            )

        if stack_session:
            sliders = ComponentGroup(
                [session_input, slice_input],
                layout=StackInCard(),
            )
            mri_input = InputGroup([view_input, sliders])
            comps = [graph, view_input, sliders]

        else:
            mri_input = InputGroup([view_input, session_input, slice_input])
            comps = [graph, view_input, slice_input]

        mri_model = SwitchableMriSlicesLookup(mri_data)
        graph_callback = ViewModelUpdateFactory(
            mri_input,
            graph,
            mri_model,
        )

        if layout is None:
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

            layout = StackLayout(as_col=as_col, width=width, sorter=sorter)

        super().__init__(
            comps,
            id_prefix=id_prefix,
            layout=layout,
            callbacks=[graph_callback],
        )


class SessionView(BaseComponentGroup):
    """Card with session information.

    Parameters
    ----------
    stack_session : bool
        Whether to stack session input with slice controller.
        If ``False``, assumes ``session_input`` component is created outside.
    """

    def __init__(
        self,
        hormones_df,
        session_input,
        session_info,
        id_prefix="",
        stack_session=False,
        layout=None,
    ):
        if layout is None:
            layout = StackLayout(as_col=False)

        # NB: a model of the hormones data
        session_info_model = PdDfLookup(
            df=hormones_df,
            output_keys=[elem.var_def.id for elem in session_info],
            tar=1,
        )
        session_callback = ViewModelUpdateFactory(
            session_input, session_info, session_info_model
        )

        super().__init__(
            [session_input, session_info] if stack_session else [session_info],
            id_prefix=id_prefix,
            layout=layout,
            callbacks=[session_callback],
            unnest=False,
        )


class MriExplorer(BaseComponentGroup):
    def __init__(
        self,
        mri_data,
        hormones_df,
        session_input,
        session_info,
        slice_input=None,
        graph_stack=None,
        id_prefix="",
        as_col=False,
        graph_first=True,
    ):
        mri_view = MriView(
            mri_data,
            session_input,
            slice_input=slice_input,
            graph_stack=graph_stack,
            stack_session=True,
            layout=DummyLayout(),
            as_col=as_col,
        )

        session_view = SessionView(
            hormones_df,
            session_input,
            session_info,
            stack_session=False,
            layout=DummyLayout(),
        )

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

        super().__init__(
            [mri_view, session_view],
            layout=GridLayout(width=width, sorter=sorter),
            unnest=False,
        )


class ModelBasedExplorer(BaseComponentGroup):
    def __init__(
        self, model, inputs, output, id_prefix="", postproc_pred=None, layout=None
    ):
        if layout is None:
            layout = StackLayout()

        callback = ViewModelUpdateFactory(inputs, output, model, postproc_pred)

        super().__init__(
            [inputs, output],
            id_prefix=id_prefix,
            layout=layout,
            callbacks=[callback],
            unnest=False,
        )


class ImageExplorer(ModelBasedExplorer):
    def __init__(
        self,
        model,
        inputs,
        image=None,
        id_prefix="",
        layout=None,
        as_col=True,
        image_first=False,
    ):
        if layout is None:
            if image_first:
                sorter = lambda x: list(reversed(x))
            else:
                sorter = lambda x: x
            layout = StackLayout(as_col=as_col, sorter=sorter)

        if image is None:
            image = Image(
                id_="image-expl",
                id_prefix=id_prefix,
                layout=lambda comp: html.Div(
                    comp,
                    style={"paddingTop": "0px"},
                ),
            )

        super().__init__(model, inputs, image, id_prefix=id_prefix, layout=layout)


class MeshExplorer(ModelBasedExplorer):
    def __init__(
        self, model, inputs, graph=None, id_prefix="", postproc_pred=None, layout=None
    ):
        # TODO: check need/use
        if graph is None:
            graph = Graph(id_="mesh-plot", plotter=MeshPlotter(), id_prefix=id_prefix)

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
        self,
        models,
        inputs,
        outputs,
        id_prefix="",
        postproc_pred=None,
        layout=None,
    ):
        if layout is None:
            layout = StackLayout()

        callbacks = [
            ViewModelUpdateFactory(
                inputs,
                output,
                model,
                postproc_pred=postproc_pred,
            )
            for output, model in zip(outputs, models)
        ]

        super().__init__(
            [inputs, outputs],
            id_prefix=id_prefix,
            layout=layout,
            callbacks=callbacks,
            unnest=False,
        )


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
        # TODO: add more syntax sugar? e.g. pass data
        # ignores button if only one model

        # TODO: add verifications?
        if graph is None:
            graph = Graph(id_="mesh-plot", plotter=MeshPlotter(), id_prefix=id_prefix)

        if layout is None:
            # TODO: delete
            # layout = GraphInputTwoColumnLayout()
            layout = StackLayout()

        # TODO: as output_group?
        inputs_cards = BaseComponentGroup(
            [
                HideableComponent(
                    id_=f"{index}_slider_container",
                    # TODO: this goes to the input
                    component=comp,
                )
                for index, comp in enumerate(inputs)
            ]
        )

        # NB: controls visibility of plots
        if checkbox_labels:
            # TODO: allow control of default visibility?
            n_graphs = graph.plotter.n_graphs

            checkbox_labels = [
                label
                if len(label) == 3
                else (label[0], label[1], label[0] < n_graphs - 1)
                for label in checkbox_labels
            ]
            # TODO: check id_prefix
            checklist = Checklist(
                id_="show-model-checkbox",
                checkbox_labels=checkbox_labels,
                # NB: assumes graph has plotter with n_graphs
                n_options=n_graphs,
            )
        else:
            checklist = DummyComponent()

        # TODO: will have to self.prefix_id; check need though!
        button = (
            AdaptedDashComponent(
                html.Button(
                    button_label,
                    # TODO: prefix
                    id="switch-model-button",
                    n_clicks=0,
                )
            )
            if len(models) > 1
            else DummyComponent()
        )

        callback = ButtonTogglerForViewModelUpdateFactory(
            output_view=graph,
            input_views=inputs,
            models=models,
            toggle_id=button.id,
            checklist=checklist,
            hideable_components=inputs_cards,
            postproc_pred=postproc_pred,
        )

        inputs_ = BaseComponentGroup([button, checklist, inputs_cards])

        super().__init__(
            [graph, inputs_],
            id_prefix=id_prefix,
            layout=layout,
            callbacks=[callback],
            unnest=False,
        )


class HideableComponent(IdComponent):
    def __init__(self, id_, component, id_prefix="", id_suffix=""):
        # TODO: add layout?
        super().__init__(id_, id_prefix, id_suffix)

        if not hasattr(component, "to_dash"):
            component = AdaptedDashComponent(component)

        # TODO: pass id_prefix to component?
        self.component = component

    def to_dash(self):
        if self._dash_component is not None:
            return self._dash_component

        self._dash_component = [
            html.Div(
                children=self.component.to_dash(),
                id=self.id,
            )
        ]

        return self._dash_component

    def as_input(self):
        return self.component.as_input()


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
        if self._dash_component is not None:
            return self._dash_component

        self._dash_component = [
            dcc.Checklist(
                id=self.id,
                options=self.options,
                value=self.as_value(self._default_bool_vis),
                inline=self.inline,
            )
        ]
        return self._dash_component

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
        if self._dash_component is not None:
            return self._dash_component

        self._dash_component = [
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
        return self._dash_component


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
        if self._dash_component is not None:
            return self._dash_component

        self._dash_component = self.func(**self.kwargs)

        return self._dash_component


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


class InputGroup:
    # to distinguish between components with to_dash
    # separates location from behavior
    # useful to create callbacks

    def __init__(self, components):
        self.components = components

    def as_input(self):
        return unnest_list(component.as_input() for component in self.components)
