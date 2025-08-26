import os
import sys

import dash_bootstrap_components as dbc
from dash import Dash

from polpo.dash.components import ImageExplorer, Slider
from polpo.dash.layout import StackLayout
from polpo.dash.style import update_style
from polpo.dash.utils import load_asset_images
from polpo.dash.variables import VarDef
from polpo.models import ListLookup


def _create_layout(assets_folder, as_col, image_first):
    images = load_asset_images(f"{assets_folder}{os.path.sep}digits")

    # TODO: do version with DictLookup
    model = ListLookup(images)

    digits = VarDef(id_="digitsID", name="Digits", min_value=0, max_value=9)
    # TODO: improve slider for spacing to label when not as_col
    inputs = Slider(digits)

    image_seq_explorer = ImageExplorer(
        model,
        inputs,
        layout=StackLayout(as_col=as_col, reverse=not image_first),
    )
    if as_col:
        return dbc.Container(image_seq_explorer.to_dash())

    return StackLayout(width=3)(image_seq_explorer.to_dash())


def my_app(as_col=True, image_first=False, run=True):
    style = {
        "margin_side": "20px",
        "text_fontsize": "24px",
        "text_fontfamily": "Avenir",
        "title_fontsize": "40px",
        "space_between_sections": "70px",
        "space_between_title_and_content": "30px",
    }
    update_style(style)

    assets_folder = "./image_sequence_explorer"

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=False,
        assets_folder=assets_folder,
    )

    layout = _create_layout(assets_folder, as_col, image_first)

    app.layout = layout

    if run:
        app.run(
            debug=True,
            use_reloader=False,
            host="0.0.0.0",
            port="8050",
        )

    return app
