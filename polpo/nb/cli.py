import logging
import sys
from typing import List

import nbformat
import typer

from polpo.nb.utils import (
    add_metadata_values,
    get_broken_nb_links,
    get_broken_nb_local_links,
    get_metadata_keys,
    get_metadata_values,
    get_nb_local_links,
    get_run_stats,
    metadata_key_contains_val,
    remove_metadata_key,
    remove_metadata_values,
    rename_metadata_key,
)
from polpo.utils import expand_path_names

app = typer.Typer()


def _set_logger(logging_level):
    logging.basicConfig(
        level=logging_level, format="%(levelname)s:polpo-nb:%(message)s"
    )


class NbItem:
    def __init__(self, nb, path):
        self.nb = nb
        self.path = path
        self._modified = False

    @property
    def modified(self):
        return self._modified

    @modified.setter
    def modified(self, val):
        if val:
            logging.info(f"Modified {self.path.as_posix()}.")

        self._modified = val

    @classmethod
    def from_path(cls, path):
        nb = nbformat.read(path, as_version=4)
        return cls(nb, path)

    def write(self, path=None):
        modified = True
        if path is None:
            path = self.path
            modified = self.modified

        if modified:
            nbformat.write(self.nb, path)

        return path


def _expand_notebook_names(names):
    """Expand a list of filename/glob names into a de-duplicated list of Paths.

    Notes
    -----
    * shell resolves globs before they reach here
    """
    paths = expand_path_names(names)
    return [path for path in paths if path.suffix == ".ipynb"]


def _load_notebooks(names):
    paths = _expand_notebook_names(names)

    return [NbItem.from_path(path) for path in paths]


def _write_notebooks(notebooks):
    for notebook in notebooks:
        notebook.write()


def _process_cs_string(string):
    if "," in string:
        string = string.replace(" ", "").split(",")
    else:
        string = [string]

    return string


def _collection_as_str(data, sep=","):
    return sep.join(data)


def _modify_with_vals(notebooks, vals, key, func):
    notebooks = _load_notebooks(notebooks)

    vals = _process_cs_string(vals)

    for notebook in notebooks:
        notebook.modified = func(notebook.nb, vals, key=key)

    _write_notebooks(notebooks)


def _nbitem_ls_to_dict(notebooks):
    return {nb.path: nb.nb for nb in notebooks}


def _write_dict_iter_stdout(data, tab_size=3, sep=","):
    text = ""
    tab = " " * tab_size
    for key, vals in data.items():
        text += f"{key}\n{tab}"
        text += _collection_as_str(vals, sep=sep + tab)
        text += "\n\n"

    sys.stdout.write(text)


def _write_dict_stdout(data, sep="\n"):
    text = sep.join([nb.path.as_posix() for nb in data])

    if text == "":
        return

    text += "\n"

    sys.stdout.write(text)


@app.command()
def rm_md_key(
    notebooks: List[str],
    key: str,
    logging_level: int = 20,
):
    _set_logger(logging_level)

    notebooks = _load_notebooks(notebooks)

    for notebook in notebooks:
        notebook.modified = remove_metadata_key(notebook.nb, key)

    _write_notebooks(notebooks)


@app.command()
def mv_md_key(
    notebooks: List[str],
    old_key: str,
    new_key: str,
    logging_level: int = 20,
):
    _set_logger(logging_level)

    notebooks = _load_notebooks(notebooks)

    for notebook in notebooks:
        notebook.modified = rename_metadata_key(notebook.nb, old_key, new_key)

    _write_notebooks(notebooks)


@app.command()
def add_md_vals(
    notebooks: List[str],
    vals: str,
    key: str = "tags",
    logging_level: int = 20,
):
    _set_logger(logging_level)

    _modify_with_vals(notebooks, vals, key, add_metadata_values)


@app.command()
def rm_md_vals(
    notebooks: List[str],
    vals: str,
    key: str = "tags",
    logging_level: int = 20,
):
    _set_logger(logging_level)

    _modify_with_vals(notebooks, vals, key, remove_metadata_values)


@app.command()
def get_md_vals(
    notebooks: List[str],
    key: str = "tags",
    tab_size: int = 3,
    sep: str = "\n",
):
    notebooks = _load_notebooks(notebooks)

    data = {}
    for notebook in notebooks:
        key_vals = get_metadata_values(notebook.nb, key=key)
        if key_vals is not None:
            data[notebook.path] = key_vals

    _write_dict_iter_stdout(data, tab_size=tab_size, sep=sep)


@app.command()
def get_md_keys(
    notebooks: List[str],
    tab_size: int = 3,
    exclude: str = "kernelspec,language_info",
    sep: str = "\n",
):
    notebooks = _load_notebooks(notebooks)
    exclude = set(_process_cs_string(exclude))

    data = {}
    for notebook in notebooks:
        keys = get_metadata_keys(notebook.nb)
        filtered_keys = [key for key in keys if key not in exclude]
        if filtered_keys:
            data[notebook.path] = filtered_keys

    _write_dict_iter_stdout(data, tab_size=tab_size, sep=sep)


@app.command()
def mdkey_contains_val(
    notebooks: List[str],
    val: str,
    key: str = "tags",
    sep: str = "\n",
):
    notebooks = _load_notebooks(notebooks)

    filtered = filter(
        lambda notebook: metadata_key_contains_val(notebook.nb, val, key=key),
        notebooks,
    )

    _write_dict_stdout(filtered, sep=sep)


@app.command()
def has_mdkey(
    notebooks: List[str],
    key: str = "tags",
    sep: str = "\n",
):
    notebooks = _load_notebooks(notebooks)

    filtered = filter(
        lambda notebook: key in get_metadata_keys(notebook.nb),
        notebooks,
    )

    _write_dict_stdout(filtered, sep=sep)


@app.command()
def get_nb_local_links(
    notebooks: List[str],
    tab_size: int = 3,
    sep: str = "\n",
    broken: bool = False,
):
    notebooks = _load_notebooks(notebooks)

    data = {}

    if broken:
        data = get_broken_nb_local_links(_nbitem_ls_to_dict(notebooks))
    else:
        for notebook in notebooks:
            links = get_nb_local_links(notebook.nb)
            if links:
                data[notebook.path] = links

    _write_dict_iter_stdout(data, tab_size=tab_size, sep=sep)


@app.command()
def get_nb_links(
    notebooks: List[str],
    tab_size: int = 3,
    sep: str = "\n",
    broken: bool = False,
):
    notebooks = _load_notebooks(notebooks)

    func = get_broken_nb_links if broken else get_nb_links

    data = {}
    for notebook in notebooks:
        links = func(notebook.nb)
        if links:
            data[notebook.path] = links

    _write_dict_iter_stdout(data, tab_size=tab_size, sep=sep)


@app.command()
def get_run_status(
    notebooks: List[str],
    tab_size: int = 3,
    sep: str = "\n",
):
    notebooks = _load_notebooks(notebooks)

    data = {"norun": [], "partial": []}
    for notebook in notebooks:
        null_cells, code_cells = get_run_stats(notebook.nb)

        if null_cells == code_cells:
            data["norun"].append(notebook.path.as_posix())

        elif null_cells:
            data["partial"].append(notebook.path.as_posix())

    for key in ("norun", "partial"):
        if len(data[key]) == 0:
            del data[key]

    _write_dict_iter_stdout(data, tab_size=tab_size, sep=sep)
