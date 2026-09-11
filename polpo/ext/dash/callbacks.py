from dash import Input, Output, callback, ctx, html

from polpo.utils import unnest_list


def create_view_model_update(
    input_view,
    output_view,
    model,
    allow_duplicate=False,
    prevent_initial_call=False,
    postproc_pred=None,
):
    empty_output = output_view.as_empty_output()

    @callback(
        output_view.as_output(allow_duplicate=allow_duplicate),
        *input_view.as_input(),
        prevent_initial_call=prevent_initial_call,
    )
    def view_model_update(*args):
        if args[0] is None:
            return empty_output

        pred = model.predict(args)
        if postproc_pred is not None:
            pred = postproc_pred(pred)

        return output_view.to_dash(pred)


def create_button_toggler(toggle_id, hideable_components):
    n_components = len(hideable_components)

    @callback(
        unnest_list(
            [
                component.as_output(component_property="style")
                for component in hideable_components
            ]
        ),
        Input(toggle_id, "n_clicks"),
    )
    def button_toggler(n_clicks):
        show_index = n_clicks % n_components

        out = [{"display": "none"}] * n_components
        out[show_index] = {"display": "block"}

        return out


def create_button_toggler_for_view_model_update(
    input_views,
    output_view,
    models,
    checklist,
    hideable_components,
    toggle_id=None,
    postproc_pred=None,
):
    """

    Parameters
    ----------
    input_views : list[Component]
    output_view : Component
    models : list[Model]
        One model per input view.
    checklist : Checklist or DummyComponent
    hideable_components : list[HideableComponent]
    models : list[Model]
    """
    # TODO: connect input view with hideable component?

    # NB: a simple merge of the two above to avoid chained callbacks
    empty_output = output_view.as_empty_output()

    n_components = len(hideable_components)

    inputs = []
    indices = [0]
    for input_view in input_views:
        input_ = input_view.as_input()
        indices.append(indices[-1] + len(input_))
        inputs.extend(input_)

    @callback(
        unnest_list(
            [
                component.as_output(component_property="style")
                for component in hideable_components
            ]
        )
        + output_view.as_output(),
        *(
            ([Input(toggle_id, "n_clicks")] if toggle_id else [])
            + inputs
            + checklist.as_input()
        ),
    )
    def button_toggler(*args):
        if toggle_id:
            n_clicks, *args = args
        else:
            n_clicks = 1

        out = [{"display": "none"}] * n_components

        if args[0] is None:
            return out + empty_output

        show_index = n_clicks % n_components
        model = models[show_index]
        model_args = args[indices[show_index] : indices[show_index + 1]]

        out[show_index] = {"display": "block"}

        if ctx.triggered_id == checklist.id:
            pred = checklist.as_bool(args[-1])
        else:
            pred = model.predict(model_args)
            if postproc_pred is not None:
                pred = postproc_pred(pred)

        return out + output_view.to_dash(pred)


class PageRegister:
    def __init__(self):
        self.pages = {}

        self.create_callback()

    def add_page(self, path, page):
        self.pages[path] = page

    def create_callback(self):
        @callback(Output("page-content", "children"), [Input("url", "pathname")])
        def render_page_content(pathname):
            """Render the page content based on the URL."""
            page = self.pages.get(pathname)
            if page is not None:
                return page

            # If the user tries to reach a different page, return a 404 message
            return html.Div(
                [
                    html.H1("404: Not found", className="text-danger"),
                    html.Hr(),
                    html.P(f"The pathname {pathname} was not recognised..."),
                ],
                className="p-3 bg-light rounded-3",
            )

        return render_page_content
