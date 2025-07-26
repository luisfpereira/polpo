import pytest


# TODO: adapt here, do not want all the combinations
@pytest.mark.parametrize("colorized", [False])
@pytest.mark.parametrize("hormones", [True])
@pytest.mark.parametrize("week", [True])
@pytest.mark.parametrize("overlay", [False])
@pytest.mark.parametrize("hideable", [False])
# @pytest.mark.parametrize("data", ["hipp", "maternal", "multiple"])
@pytest.mark.parametrize("data", ["hipp"])
def test_mesh_explorer(dash_duo, data, hideable, overlay, week, hormones, colorized):
    from polpo.dash.app.mesh_explorer import my_app

    dash_duo.start_server(
        my_app(data, hideable, overlay, week, hormones, colorized, run=False)
    )
    dash_duo.wait_for_page(timeout=3)
    assert dash_duo.get_logs() == [], "Browser console has errors"


@pytest.mark.parametrize("column", [True, False])
@pytest.mark.parametrize("swapped", [True, False])
def test_image_app(dash_duo, column, swapped):
    from polpo.dash.app.image_explorer import my_app

    dash_duo.start_server(my_app(column, swapped, run=False))
    dash_duo.wait_for_page(timeout=3)
    assert dash_duo.get_logs() == [], "Browser console has errors"
