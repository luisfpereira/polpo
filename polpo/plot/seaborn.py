"""Utilities built on top of seaborn for embedding plots in Polpo figures."""

import seaborn as sns
from matplotlib.lines import Line2D


def plot_pairmatrix(
    subfig,
    data,
    columns,
    groupby=None,
    colors=None,
    title=None,
    diag_kind="kde",
):
    """Plot a lower-triangular pair matrix inside a Matplotlib subfigure.

    Unlike seaborn.pairplot, which manages its own figure, this function
    supports embedding pair matrices within larger figure layouts.
    """
    n = len(columns)
    axes = subfig.subplots(n, n, squeeze=False)

    kwargs = dict(data=data, hue=groupby, palette=colors, legend=False)

    for i, y in enumerate(columns):
        for j, x in enumerate(columns):
            ax = axes[i, j]

            if i < j:
                ax.set_visible(False)
            elif i == j:
                if diag_kind == "kde":
                    sns.kdeplot(x=x, ax=ax, fill=True, alpha=0.4, **kwargs)
                elif diag_kind == "hist":
                    sns.histplot(x=x, ax=ax, alpha=0.4, **kwargs)
                else:
                    raise ValueError(f"Unknown diag_kind: {diag_kind}")

                ax.set_ylabel("")

            else:
                sns.scatterplot(x=x, y=y, ax=ax, alpha=0.7, **kwargs)

            ax.label_outer()

    if title is not None:
        subfig.suptitle(title)

    return axes


def pairmatrix_legend_handles(groups, colors, marker="o"):
    """Create legend handles for grouped pair-matrix plots."""
    return [
        Line2D(
            [],
            [],
            color=colors[group],
            marker=marker,
            linestyle="none",
            label=group,
        )
        for group in groups
    ]
