import dash_bootstrap_components as dbc
from dash import Dash

import polpo.preprocessing.pd as ppd
from polpo.dash.components import (
    BaseComponentGroup,
    ComponentGroup,
    DepVar,
    SessionView,
    Slider,
    SwitchableMriView,
)
from polpo.dash.layout import (
    MultiColLayout,
    OneColMultiRowLayout,
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
        layout=lambda comps: dbc.Card(comps),
    )


def _create_layout(with_session):
    # session_input is in a column with session_info
    mri_data = _load_mri_data()

    # session controller
    session_id = VarDef(
        id_="sessionID",
        name="Session Number",
        min_value=1,
        max_value=len(mri_data),
        default_value=1,
    )
    session_input = Slider(var_def=session_id)

    if not with_session:
        mri_view = SwitchableMriView(mri_data, session_input)

        return dbc.Container(mri_view.to_dash())

    hormone_df = _load_homornes_df()
    session_info = _create_session_info(session_id)

    def ignore_session(comps):
        comps = comps.copy()
        comps.pop(2)

        return comps

    mri_view = SwitchableMriView(
        mri_data,
        session_input,
        layout=OneColMultiRowLayout(
            sorter=ignore_session,
        ),
        stack_session=False,
    )
    session_view = SessionView(hormone_df, session_input, session_info)

    mri_explorer = BaseComponentGroup(
        [mri_view, session_view], layout=MultiColLayout(width=5, sm=None)
    )

    return dbc.Container(mri_explorer.to_dash())


def my_app(with_session=False):
    style = {
        "margin_side": "20px",
        "text_fontsize": "24px",
        "text_fontfamily": "Avenir",
        "title_fontsize": "40px",
        "space_between_sections": "70px",
        "space_between_title_and_content": "30px",
    }
    update_style(style)

    layout = _create_layout(with_session)

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
