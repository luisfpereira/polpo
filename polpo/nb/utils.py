import re

from polpo.nb.links import are_links_ok


def remove_metadata_key(notebook, key):
    if key in notebook.metadata:
        notebook.metadata.pop(key)
        return True

    return False


def rename_metadata_key(notebook, old_key, new_key):
    metadata = notebook.metadata
    if old_key in metadata:
        metadata[new_key] = notebook.metadata.pop(old_key)
        return True

    return False


def add_metadata_values(notebook, values, key="tags"):
    metadata = notebook.metadata

    existing_values = metadata.get(key, [])

    modified = False
    for value in values:
        if value not in existing_values:
            modified = True
            existing_values.append(value)

    metadata[key] = existing_values

    return modified


def remove_metadata_values(notebook, values, key="tags"):
    metadata = notebook.metadata

    if key not in metadata:
        return False

    existing_values = metadata.get(key)

    modified = False

    for value in values:
        if value in existing_values:
            modified = True
            existing_values.remove(value)

    if len(existing_values) == 0:
        metadata.pop(key)
    else:
        metadata[key] = existing_values

    return modified


def get_metadata_values(notebook, key="tags"):
    metadata = notebook.metadata
    return metadata.get(key, None)


def get_metadata_keys(notebook):
    metadata = notebook.metadata
    return list(metadata.keys())


def metadata_key_contains_val(notebook, val, key="tags"):
    metadata = notebook.metadata

    if key not in metadata:
        return False

    return val in metadata.get(key)


def get_cells_by_type(notebook, cell_type="markdown"):
    return [cell for cell in notebook.cells if cell.cell_type == cell_type]


def merge_cell_sources(cells, sep="\n"):
    return f"{sep}".join([cell.source for cell in cells])


def get_source_matches(notebook, finder, unique=True):
    cells = get_cells_by_type(notebook)
    text = merge_cell_sources(cells)

    match = finder.findall(text)

    if match is None or not unique:
        return match

    return list(set(match))


def get_nb_local_links(notebook, unique=True):
    pattern = r"\((\.+/[\w\-/]+\.ipynb)\)"
    finder = re.compile(pattern)

    return get_source_matches(notebook, finder, unique=unique)


def get_nb_links(notebook, unique=True):
    pattern = r"\((https://[^\s)]+)\)"
    finder = re.compile(pattern)

    return get_source_matches(notebook, finder, unique=unique)


def get_broken_nb_local_links(notebooks_dict):
    broken_links = {}

    all_paths = set(path.resolve() for path in notebooks_dict.keys())

    for path, notebook in notebooks_dict.items():
        links_nb = get_nb_local_links(notebook, unique=True)

        broken_links_nb = [
            link for link in links_nb if (path.parent / link).resolve() not in all_paths
        ]

        if broken_links_nb:
            broken_links[path] = broken_links_nb

    return broken_links


def get_broken_nb_links(notebook):
    urls = get_nb_links(notebook)

    return [url for url, ok in zip(urls, are_links_ok(urls)) if not ok]


def get_run_stats(notebook):
    code_cells = get_cells_by_type(notebook, "code")
    null_cells = list(filter(lambda x: x["execution_count"] is None, code_cells))

    return len(null_cells), len(code_cells)
