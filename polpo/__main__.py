import logging
from enum import Enum

import typer

app = typer.Typer()


class DataOptions(str, Enum):
    hipp = "hipp"
    maternal = "maternal"
    multiple = "multiple"


@app.command()
def mesh_explorer(
    data: DataOptions = DataOptions.hipp,
    hideable: bool = False,
    overlay: bool = False,
    week: bool = True,
    hormones: bool = True,
    colorized: bool = True,
    logging_level: int = 20,
):
    """Launch mesh explorer app."""
    from polpo.dash.app.mesh_explorer import my_app

    logging.basicConfig(level=logging_level)

    data = data.value

    if hideable and data != "multiple":
        # TODO: extend to single model?
        raise ValueError(
            "Cannot handle hiddeable for single structure and multiple models."
        )

    my_app(
        data=data,
        hideable=hideable,
        overlay=overlay,
        week=week,
        hormones=hormones,
        colorized=colorized,
    )


@app.command()
def mri_explorer(
    session_view: bool = True,
    as_col: bool = False,
    graph_first: bool = True,
    switchable: bool = False,
    logging_level: int = 20,
):
    """Launch mri explorer app."""
    if not switchable:
        from polpo.dash.app.mri_explorer import my_app
    else:
        from polpo.dash.app.switchable_mri_explorer import my_app

    logging.basicConfig(level=logging_level)

    my_app(session_view, as_col, graph_first)


@app.command()
def image_explorer(
    as_col: bool = True,
    image_first: bool = False,
    logging_level: int = 20,
):
    """Launch image sequence explorer app."""
    from polpo.dash.app.image_explorer import my_app

    logging.basicConfig(level=logging_level)

    my_app(as_col, image_first)


@app.command()
def multi_image_explorer(
    logging_level: int = 20,
):
    """Launch image sequence explorer app."""
    from polpo.dash.app.multi_image_explorer import my_app

    logging.basicConfig(level=logging_level)

    my_app()


if __name__ == "__main__":
    app()
