import os

import nbformat


def _create_symlinks(notebooks_path, docs_notebook_path, renamings=None):
    # TODO: add exclude
    if renamings is None:
        renamings = {}

    for path in notebooks_path.rglob("*.ipynb"):
        if "exclude" in nbformat.read(path, as_version=4).metadata.get("nbsymlink", {}):
            continue

        rel_path = path.relative_to(notebooks_path)

        dir_path = docs_notebook_path / renamings.get(
            rel_path.parent.as_posix(), rel_path.parent
        )
        dir_path.mkdir(exist_ok=True, parents=True)

        os.symlink(
            path,
            dir_path / renamings.get(rel_path.as_posix(), rel_path.name),
            target_is_directory=False,
        )


def run(app, config):
    notebooks_path = (app.confdir / config.nbsymlink_notebooks_dir).resolve()
    doc_notebooks_path = app.srcdir / "_generated/notebooks"

    _create_symlinks(
        notebooks_path, doc_notebooks_path, renamings=config.nbsymlink_renamings
    )


def setup(app):
    """Entry point for the Sphinx extension."""
    app.add_config_value("nbsymlink_notebooks_dir", "default_value", "../notebooks")
    app.add_config_value("nbsymlink_renamings", "default_value", None, types=dict)

    app.connect("config-inited", run)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
