from matplotlib import pyplot as plt

from polpo.dataset import NestedDataset
from polpo.dataset.plot import plot_nested


def plot_trajectories(
    data,
    group_col="subject",
    time_col="time",
    value_col="Y",
    facet_col=None,
    ax=None,
    xlabel="Time",
    ylabel=None,
    **kwargs,
):
    if facet_col is not None:
        return _plot_grouped_trajectories(
            data,
            facet_col=facet_col,
            group_col=group_col,
            time_col=time_col,
            value_col=value_col,
            axes=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            **kwargs,
        )

    nested = NestedDataset.from_dataframe(
        data,
        outer_col=group_col,
        inner_col=time_col,
        value_col=value_col,
    )

    ax = plot_nested(
        nested,
        ax=ax,
        **kwargs,
    )

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return ax


def _plot_grouped_trajectories(
    data,
    facet_col,
    group_col,
    time_col,
    value_col,
    axes=None,
    xlabel=None,
    ylabel=None,
    **kwargs,
):
    facets = data[facet_col].unique()

    if axes is None:
        fig, axes = plt.subplots(
            1,
            len(facets),
            figsize=(4 * len(facets), 5),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes = axes.ravel()
    else:
        fig = axes[0].ravel().figure

    if len(axes) != len(facets):
        raise ValueError(f"Expected {len(facets)} axes, got {len(axes)}.")

    for ax, facet in zip(axes, facets):
        plot_trajectories(
            data[data[facet_col] == facet],
            group_col=group_col,
            time_col=time_col,
            value_col=value_col,
            ax=ax,
            xlabel=None,
            ylabel=None,
            **kwargs,
        )
        ax.set_title(str(facet))

    if xlabel is not None:
        fig.supxlabel(xlabel)

    if ylabel is not None:
        fig.supylabel(ylabel)

    fig.subplots_adjust(wspace=0.05)

    return axes
