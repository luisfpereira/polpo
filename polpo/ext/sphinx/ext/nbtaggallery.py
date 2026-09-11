import nbformat
from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles

from polpo.sphinx.utils import create_directive_str


class NBTagGalleries(SphinxDirective):
    has_content = False
    required_arguments = 0
    option_spec = {
        "path": directives.path,
        "auto": directives.flag,
    }

    def run(self):
        if "path" not in self.options:
            raise self.error("NBTagGalleries: option ':path:' is required.")

        env = self.state.document.settings.env
        app = env.app
        docname = env.docname
        config = app.config

        directive_path = (app.srcdir / docname).parent

        rst = _create_galleries(
            (directive_path / self.options["path"]).resolve(),
            directive_path=directive_path,
            tags=config.nbtaggallery_tags,
            auto="auto" in self.options,
            tag_captions=config.nbtaggallery_tag_captions,
        )

        src, base = self.get_source_info()
        buf = StringList()
        for i, line in enumerate(rst.splitlines()):
            buf.append(line, src, base + i)

        container = nodes.Element()
        nested_parse_with_titles(self.state, buf, container)
        return container.children


def _group_by_tags(notebooks_path, directive_path, tags=(), auto=False):
    # directive_path for relative paths
    grouped_notebooks = {tag: [] for tag in tags}

    for file in notebooks_path.rglob("*.ipynb"):
        metadata = nbformat.read(file, as_version=4).metadata

        for tag in metadata.get("tags", []):
            if tag not in grouped_notebooks:
                if not auto:
                    continue

                grouped_notebooks[tag] = {}

            grouped_notebooks[tag].append(
                file.relative_to(directive_path, walk_up=True).as_posix()
            )

    return grouped_notebooks


def _create_galleries(
    notebooks_path, directive_path, tags=(), auto=False, tag_captions=None
):
    if tag_captions is None:
        tag_captions = {}

    grouped_notebooks = _group_by_tags(
        notebooks_path,
        directive_path,
        tags=tags,
        auto=auto,
    )

    galleries_str_ls = []
    for tag, filenames in grouped_notebooks.items():
        if len(filenames) == 0:
            continue

        caption = tag_captions.get(tag)
        if caption is None:
            caption = tag.replace("_", " ").capitalize()

        text = caption + f"\n{'='*len(caption)}"
        text += "\n\n"

        text += create_directive_str("nblinkgallery", filenames)

        text += "\n\n"
        text += create_directive_str(
            "toctree",
            filenames,
            caption=caption,
            hidden="",
            maxdepth=1,
        )
        galleries_str_ls.append(text)

    galleries_str = "\n\n\n".join(galleries_str_ls)

    return galleries_str


def setup(app):
    """Entry point for the Sphinx extension."""
    app.add_config_value("nbtaggallery_tags", "default_value", (), types=list)
    app.add_config_value("nbtaggallery_tag_captions", "default_value", None, types=dict)

    app.add_directive("nbtaggalleries", NBTagGalleries)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
