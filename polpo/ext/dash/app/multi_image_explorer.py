import os
import sys

import dash_bootstrap_components as dbc
from dash import Dash, get_asset_url

from polpo.dash.components import Image, SharedInputModelsBasedExplorer, Slider
from polpo.dash.style import update_style
from polpo.dash.variables import VarDef
from polpo.models import ListLookup
from polpo.preprocessing import Sorter
from polpo.preprocessing.path import FileFinder


def _load_images(assets_folder):
    # assumes assets at app folder level
    file_path = os.path.dirname(sys.modules[__package__].__file__)
    # removes ./
    short_assets_folder = "/".join(assets_folder.split("/")[1:])

    assets_folder_abs = os.path.join(file_path, short_assets_folder)

    images = (
        FileFinder(data_dir=os.path.join(assets_folder_abs, "digits")) + Sorter()
    )()

    n_path_assets = len(assets_folder_abs)
    return [get_asset_url(image[n_path_assets + 1 :]) for image in images]


def _create_layout(assets_folder):
    images = _load_images(assets_folder)

    # TODO: do version with DictLookup
    models = [ListLookup(images), ListLookup(images[::-1])]

    digits = VarDef(id_="digitsID", name="Digits", min_value=0, max_value=9)
    inputs = Slider(digits)

    image_style = {"width": "50%"}
    outputs = [
        Image(id_=f"image-expl-{index}", style=image_style)
        for index in range(len(models))
    ]

    image_seq_explorer = SharedInputModelsBasedExplorer(models, inputs, outputs)
    return dbc.Container(image_seq_explorer.to_dash())


def my_app():
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

    layout = _create_layout(assets_folder)

    app.layout = layout

    app.run(
        debug=True,
        use_reloader=False,
        host="0.0.0.0",
        port="8050",
    )
