import subprocess
import tempfile
from pathlib import Path


def join_table_rows(rows):
    """Join LaTeX table rows with line breaks.

    Parameters
    ----------
    rows : sequence of str
        Table rows.

    Returns
    -------
    text : str
        Rows separated by LaTeX line breaks.
    """
    return "\\\\\n".join(rows)


def add_table(table_name):
    """Generate LaTeX code to include a table.

    Parameters
    ----------
    table_name : str
        Path to the LaTeX table file.

    Returns
    -------
    text : str
        LaTeX table environment containing an input command.
    """
    table_template = r"""\begin{{table}}[H]
\centering
\input{{{table_name}}}
\end{{table}}"""
    return table_template.format(table_name=table_name)


def add_figure(figure_name, width=1.0):
    """Generate LaTeX code to include a figure.

    Parameters
    ----------
    figure_name : str
        Path to the figure.
    width : float
        Figure width as a fraction of the line width.

    Returns
    -------
    text : str
        LaTeX figure environment.
    """
    per_subject_fig_template = r"""\begin{{figure}}[H]
\centering
\includegraphics[width={width}\linewidth]{{{figure_name}}}
\end{{figure}}"""
    return per_subject_fig_template.format(
        figure_name=figure_name,
        width=width,
    )


def add_subfigure(figure_name, width=1.0):
    """Generate LaTeX code to include a subfigure.

    Parameters
    ----------
    figure_name : str
        Path to the figure.
    width : float
        Subfigure width as a fraction of the text width.

    Returns
    -------
    text : str
        LaTeX subfigure environment.
    """
    template = r"""\begin{{subfigure}}{{{width}\textwidth}}
\centering
\includegraphics[width=\linewidth]{{{figure_name}}}
\end{{subfigure}}"""
    return template.format(
        figure_name=figure_name,
        width=width,
    )


def add_subfigures(
    figure_names,
    width=None,
    n_cols=2,
    max_rows=None,
    caption=None,
):
    """Arrange figures in a LaTeX subfigure grid.

    Figures exceeding the maximum number of rows are placed in
    continued figure environments.

    Parameters
    ----------
    figure_names : sequence of str
        Paths to the figures, in display order.
    width : float or None
        Subfigure width as a fraction of the text width.
        If None, determined from the number of columns.
    n_cols : int
        Number of columns per figure environment.
    max_rows : int or None
        Maximum number of rows per figure environment.
        If None, all figures are placed in one environment.
    caption : str or None
        Caption included in each figure environment.

    Returns
    -------
    text : str
        LaTeX code containing the arranged subfigures.
    """
    if width is None:
        width = round(0.95 / n_cols, 2)

    figs_per_page = len(figure_names) if max_rows is None else n_cols * max_rows

    caption = "" if caption is None else f"\n\\caption{{{caption}}}"

    template = r"""\begin{{figure}}[H]{continued}
\centering
{rows}{caption}
\end{{figure}}"""

    figures = []

    for start in range(0, len(figure_names), figs_per_page):
        names = figure_names[start : start + figs_per_page]

        rows = []
        for i, name in enumerate(names):
            if i > 0:
                if i % n_cols == 0:
                    rows.append("\n\n\\vspace{0.5em}\n\n")
                else:
                    rows.append("\n\\hfill\n")

            rows.append(add_subfigure(name, width))

        figures.append(
            template.format(
                continued="" if start == 0 else r"\ContinuedFloat",
                rows="".join(rows),
                caption=caption,
            )
        )

    return "\n\n".join(figures)


def add_subsection(name):
    """Generate a LaTeX subsection command.

    Parameters
    ----------
    name : str
        Subsection title.

    Returns
    -------
    text : str
        LaTeX subsection command.
    """
    return r"""\subsection{{{name}}}""".format(name=name)


def make_latex_document(body, packages=None, preamble=None):
    """Generate a standalone LaTeX article document.

    Parameters
    ----------
    body : str
        Document content.
    packages : sequence of str or None
        LaTeX packages to include.
    preamble : str or None
        Additional LaTeX preamble content.

    Returns
    -------
    text : str
        Complete LaTeX document source.
    """
    if packages is None:
        packages = []

    packages_text = "\n".join(rf"\usepackage{{{package}}}" for package in packages)

    preamble = "" if preamble is None else preamble

    return rf"""\documentclass{{article}}

{packages_text}
{preamble}

\begin{{document}}
{body}
\end{{document}}
"""


def make_figure_document(body, figures_path):
    """Generate a standalone LaTeX document for figures.

    Includes the packages required for figures and subfigures,
    configures the graphics directory, and disables page numbering.

    Parameters
    ----------
    body : str
        LaTeX figure content.
    figures_path : pathlib.Path
        Directory containing the figure files.

    Returns
    -------
    text : str
        Complete LaTeX document source.
    """
    preamble = rf"""
\graphicspath{{{{{figures_path.resolve()}/}}}}
\pagestyle{{empty}}
"""

    return make_latex_document(
        body,
        packages=["graphicx", "float", "caption", "subcaption"],
        preamble=preamble,
    )


def compile_latex(text, filename):
    """Compile LaTeX source into a PDF using pdflatex.

    Compilation takes place in a temporary directory.

    Parameters
    ----------
    text : str
        Complete LaTeX document source.
    filename : pathlib.Path
        Destination path for the generated PDF.

    Raises
    ------
    subprocess.CalledProcessError
        If LaTeX compilation fails.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tex_path = tmpdir / "document.tex"
        tex_path.write_text(text)

        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=tmpdir,
            check=True,
        )

        filename.write_bytes((tmpdir / "document.pdf").read_bytes())
