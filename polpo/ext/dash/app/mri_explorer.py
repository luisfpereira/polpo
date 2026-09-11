import dash_bootstrap_components as dbc
from dash import Dash

import polpo.preprocessing.pd as ppd
from polpo.dash.components import (
    ComponentGroup,
    DepVar,
    MriExplorer,
    MriSliders,
    Slider,
)
from polpo.dash.style import update_style
from polpo.dash.variables import VarDef
from polpo.preprocessing import Map, Pipeline, Sorter, Truncater
from polpo.preprocessing.load.pregnancy import (
    DenseMaternalCsvDataLoader,
    PregnancyPilotMriLoader,
)
from polpo.preprocessing.mri import MriImageLoader


def _load_homornes_df():
    return Pipeline(
        steps=[
            DenseMaternalCsvDataLoader(pilot=True),
            ppd.Drop(labels=27),
        ]
    )()


def _load_mri_data():
    return Pipeline(
        steps=[
            PregnancyPilotMriLoader(as_dict=False),
            Sorter(),
            Truncater(value=2),  # For debugging
            Map(step=MriImageLoader(), n_jobs=1, verbose=1),
        ]
    )()


def _create_inputs(session_id):
    mri_x = VarDef(id_="mri_x", name="X Coordinate (Changes Side View)")
    mri_y = VarDef(id_="mri_y", name="Y Coordinate (Changes Front View)")
    mri_z = VarDef(id_="mri_z", name="Z Coordinate (Changes Top View)")

    return MriSliders(
        components=[
            Slider(var_def=session_id),
            Slider(var_def=mri_x, step=5),
            Slider(var_def=mri_y, step=5),
            Slider(var_def=mri_z, step=5),
        ],
        trims=[[20, 40], 50, 70],
    )


def _create_session_info(session_id):
    estro = VarDef(
        id_="estro", name="Estrogen", unit="pg/ml", min_value=4100, max_value=12400
    )
    lh = VarDef(id_="lh", name="LH", unit="ng/ml", min_value=0.59, max_value=1.45)
    gest_week = VarDef(
        id_="gestWeek",
        name="Gestational Week",
        min_value=0,
        max_value=36,
        default_value=15,
    )
    endo_status = VarDef(id_="endoStatus", name="Pregnancy status")
    trimester = VarDef(id_="trimester", name="Trimester")

    return ComponentGroup(
        title="Session Information",
        components=[
            DepVar(var_def=session_id),
            DepVar(var_def=gest_week),
            DepVar(var_def=estro),
            DepVar(var_def=lh),
            DepVar(var_def=endo_status),
            DepVar(var_def=trimester),
        ],
    )


def _create_layout():
    mri_data = _load_mri_data()
    hormone_df = _load_homornes_df()

    session_id = VarDef(
        id_="sessionID", name="Session Number", min_value=1, max_value=26
    )

    sliders = _create_inputs(session_id)
    session_info = _create_session_info(session_id)

    mri_explorer = MriExplorer(mri_data, hormone_df, sliders, session_info)

    return dbc.Container(mri_explorer.to_dash())


def my_app(
    data="hipp",
    hideable=False,
    overlay=False,
    week=True,
    hormones=True,
):
    style = {
        "margin_side": "20px",
        "text_fontsize": "24px",
        "text_fontfamily": "Avenir",
        "title_fontsize": "40px",
        "space_between_sections": "70px",
        "space_between_title_and_content": "30px",
    }
    update_style(style)

    layout = _create_layout()

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    app.layout = layout

    app.run(
        debug=True,
        use_reloader=False,
        host="0.0.0.0",
        port="8050",
    )
