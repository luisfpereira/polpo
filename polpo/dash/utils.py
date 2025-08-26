import inspect
import os

from dash import get_asset_url

from polpo.preprocessing import Map, Sorter
from polpo.preprocessing.path import FileFinder, PathShortener


def load_asset_images(assets_folder, app_path=None):
    """Load images.

    Parameters
    ----------
    app_path : str
        e.g. ``os.path.dirname(sys.modules[__package__].__file__)``.
        Default assumes assets at app folder level.
    assets_folder :
        Relative path from app to assets folder.
    """
    if app_path is None:
        app_path = PathShortener(init_index=0, last_index=-1)(
            inspect.stack()[1].filename
        )

    if "." in assets_folder:
        # removes ./
        assets_folder = "/".join(assets_folder.split("/")[1:])

    assets_folder_abs = os.path.join(app_path, assets_folder)

    images = (
        FileFinder()
        + Sorter()
        + Map(
            PathShortener(init_index=len(app_path.split(os.path.sep)) + 1)
            + get_asset_url
        )
    )(assets_folder_abs)

    return images
