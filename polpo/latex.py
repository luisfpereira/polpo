import subprocess
import tempfile
from pathlib import Path


def join_table_rows(rows):
    return "\\\\\n".join(rows)


def add_table(table_name):
    table_template = r"""\begin{{table}}[H]
\centering
\input{{{table_name}}}
\end{{table}}"""
    return table_template.format(table_name=table_name)


def add_figure(figure_name, width=1.0):
    per_subject_fig_template = r"""\begin{{figure}}[H]
\centering
\includegraphics[width={width}\linewidth]{{{figure_name}}}
\end{{figure}}"""
    return per_subject_fig_template.format(
        figure_name=figure_name,
        width=width,
    )


def add_subfigure(figure_name, width=1.0):
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
    return r"""\subsection{{{name}}}""".format(name=name)


def make_latex_document(body, packages=None, preamble=None):
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
