import time

import pytest

# NB: tests if everything is created, nothing else


def _test_app(dash_duo, app, maximize_window=True, sleep=2):
    # TODO: control these two parameters with env vars
    dash_duo.start_server(app)

    if maximize_window:
        dash_duo.driver.maximize_window()

    # FIXME: because something weird with wait_for_page
    time.sleep(sleep)

    dash_duo.wait_for_page(timeout=3)

    assert dash_duo.get_logs() == [], "Browser console has errors"


@pytest.mark.parametrize(
    "data,hideable,overlay,week,hormones,colorized",
    [
        ("hipp", False, True, True, False, False),
        ("hipp", False, False, True, True, True),
        ("maternal", False, True, True, True, False),
        ("multiple", True, False, True, False, False),
        ("multiple", False, False, True, True, True),
    ],
)
def test_mesh_explorer(dash_duo, data, hideable, overlay, week, hormones, colorized):
    from polpo.dash.app.mesh_explorer import my_app

    _test_app(
        dash_duo,
        my_app(data, hideable, overlay, week, hormones, colorized, run=False),
    )


@pytest.mark.parametrize("switchable", [False, True])
@pytest.mark.parametrize("graph_first", [False, True])
@pytest.mark.parametrize("as_col", [False, True])
@pytest.mark.parametrize("session_view", [False, True])
def test_mri_explorer(dash_duo, session_view, as_col, graph_first, switchable):
    if not switchable:
        from polpo.dash.app.mri_explorer import my_app
    else:
        from polpo.dash.app.switchable_mri_explorer import my_app

    _test_app(
        dash_duo,
        my_app(session_view, as_col, graph_first, run=False),
    )


@pytest.mark.parametrize("multiple", [True, False])
@pytest.mark.parametrize("image_first", [True, False])
@pytest.mark.parametrize("as_col", [True, False])
def test_image_explorer(dash_duo, as_col, image_first, multiple):
    if not multiple:
        from polpo.dash.app.image_explorer import my_app
    else:
        from polpo.dash.app.multi_image_explorer import my_app

    _test_app(
        dash_duo,
        my_app(as_col, image_first, run=False),
    )
