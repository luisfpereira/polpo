import dash_bootstrap_components as dbc
from dash import Dash

import polpo.preprocessing.pd as ppd
from polpo.dash.components import (
    ComponentGroup,
    DepVar,
    MriExplorer,
    MriView,
    Slider,
)
from polpo.dash.layout import (
    MriExplorerLayout,
    StackInCard,
    StackLayout,
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
        layout=StackInCard(gap=0),
    )


def _create_layout(session_view, as_col, graph_first):
    mri_data = _load_mri_data()

    session_id = VarDef(
        id_="sessionID", name="Session Number", min_value=1, max_value=len(mri_data)
    )
    session_input = Slider(var_def=session_id)

    if not session_view:
        mri_view = MriView(
            mri_data,
            session_input=session_input,
            layout=StackLayout(as_col=as_col, reverse=not graph_first),
            graph_stack_layout=StackLayout(as_col=not as_col),
        )

        return dbc.Container(mri_view.to_dash())

    hormone_df = _load_homornes_df()
    session_info = _create_session_info(session_id)

    mri_explorer = MriExplorer(
        mri_data,
        hormone_df,
        session_input,
        session_info,
        layout=MriExplorerLayout(as_col=as_col, graph_first=graph_first),
        graph_stack_layout=StackLayout(as_col=not as_col),
    )

    return dbc.Container(mri_explorer.to_dash())


def my_app(session_view=True, as_col=False, graph_first=True, run=True):
    style = {
        "margin_side": "20px",
        "text_fontsize": "24px",
        "text_fontfamily": "Avenir",
        "title_fontsize": "40px",
        "space_between_sections": "70px",
        "space_between_title_and_content": "30px",
    }
    update_style(style)

    layout = _create_layout(session_view, as_col, graph_first)

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    app.layout = layout

    if run:
        app.run(
            debug=True,
            use_reloader=False,
            host="0.0.0.0",
            port="8050",
        )

    return app
