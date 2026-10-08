import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from .base import Plotter


class SlicesPlotter(Plotter):
    # TODO: check volume.plot
    def __init__(self, cmap="gray", vertical=False):
        super().__init__()
        self.cmap = cmap
        self.vertical = vertical

    def plot(self, slices):
        if isinstance(slices, np.ndarray):
            n_slices = 1
            slices = [slices]
        else:
            n_slices = len(slices)

        n_rows, n_cols = 1, n_slices
        if self.vertical:
            n_rows, n_cols = n_cols, n_rows

        fig, axes = plt.subplots(n_rows, n_cols, constrained_layout=True)
        if n_slices == 1:
            axes = [axes]

        for ax, slice_ in zip(axes, slices):
            ax.imshow(
                slice_.T,
                cmap=self.cmap,
                origin="lower",
            )

        return fig, axes


def _get_groups_and_subgroups(data, has_all=True):
    # has_all: whether all groups have all subgroups
    groups = list(data.keys())

    if has_all:
        subgroups = list(data[groups[0]].keys())
    else:
        subgroups = sorted(
            {subgroup for group in data.values() for subgroup in group.keys()}
        )

    return groups, subgroups


def _compute_grouped_barplot_stats(data, groups, subgroups, agg, compute_std):
    vals = []
    stds = []
    for group_index, group in enumerate(groups):
        group_vals = []
        vals.append(group_vals)
        group_stds = []
        stds.append(group_stds)

        for subgroup_index, subgroup in enumerate(subgroups):
            arr = np.asarray(data.get(group, {}).get(subgroup, []))

            group_vals.append(agg(arr))
            if compute_std:
                group_stds.append(np.std(arr))

    vals = np.asarray(vals)
    stds = np.asarray(stds)

    return vals, stds


def grouped_barplot(
    data,
    agg=None,
    show_std=True,
    cmap="tab10",
    ax=None,
    xtick_rotation=30,
):
    if agg is None:
        agg = np.mean

    if ax is None:
        _, ax = plt.subplots()

    groups, subgroups = _get_groups_and_subgroups(data, has_all=True)
    vals, stds = _compute_grouped_barplot_stats(data, groups, subgroups, agg, show_std)

    n_groups = len(groups)
    n_subgroups = len(subgroups)

    x = np.arange(n_groups)
    total_width = 0.8
    bar_width = total_width / max(n_subgroups, 1)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    colors = {subgroups[index]: cmap(index % cmap.N) for index in range(n_subgroups)}

    for subgroup_index, subgroup in enumerate(subgroups):
        offsets = x - total_width / 2 + subgroup_index * bar_width + bar_width / 2

        y = vals[:, subgroup_index]
        yerr = stds[:, subgroup_index] if show_std else None
        ax.bar(
            offsets,
            y,
            width=bar_width,
            yerr=yerr,
            label=subgroup,
            color=colors[subgroup],
            capsize=3,
            ecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=xtick_rotation)

    return ax


def plot_grid(
    data,
    plot,
    select=None,
    n_cols=2,
    legend_position="bottom",
    legend_wrap=1,
    sharex=True,
    sharey=False,
    figsize=None,
    share_legend=True,
    **kwargs,
):
    if select is None:
        select = lambda label, item: (item,)

    n_items = len(data)
    n_cols = min(n_cols, n_items)
    n_rows = math.ceil(n_items / n_cols)

    if figsize is None:
        figsize = (4.5 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        layout="constrained",
    )
    fig.get_layout_engine().set(
        h_pad=0.12,
    )

    data_axes = list(axes.flat)[:n_items]

    for index, (ax, (label, item)) in enumerate(zip(data_axes, data.items())):
        plot(*select(label, item), ax=ax, **kwargs)

        ax.set_title(label, fontsize=11, pad=10)

        row, col = divmod(index, n_cols)

        if col != 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=not sharey)

        if row != n_rows - 1:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=not sharex)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # hide unused subplots in the last row
    for ax in list(axes.flat)[n_items:]:
        ax.set_visible(False)

    if not share_legend:
        return fig, axes

    legends = [ax.get_legend() for ax in data_axes if ax.get_legend() is not None]

    if not legends:
        return fig, axes

    legend = legends[0]
    handles = legend.legend_handles
    labels = [text.get_text() for text in legend.get_texts()]

    for legend in legends:
        legend.remove()

    ncol = math.ceil(len(handles) / legend_wrap)

    if legend_position == "bottom":
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=ncol,
            frameon=False,
        )

    elif legend_position == "right":
        fig.legend(
            handles,
            labels,
            loc="outside right center",
            ncol=legend_wrap,
            frameon=False,
        )

    else:
        raise ValueError(f"Unknown legend position: {legend_position!r}")

    return fig, axes


def _legend_handle(name, kwargs):
    return Line2D(
        [],
        [],
        linestyle="none",
        marker=kwargs.get("marker", "o"),
        markerfacecolor=kwargs.get(
            "facecolors",
            kwargs.get("color", "none"),
        ),
        markeredgecolor=kwargs.get("color"),
        label=name,
    )


def plot_layers(
    data,
    plot_fnc,
    ax=None,
    layer_kwargs=None,
    legend=True,
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots()

    layer_kwargs = {} if layer_kwargs is None else layer_kwargs

    for name, item in data.items():
        kwargs_ = kwargs | layer_kwargs.get(name, {})
        plot_fnc(item, ax=ax, **kwargs_)

    if legend:
        handles = [
            _legend_handle(name, kwargs) for name, kwargs in layer_kwargs.items()
        ]
        ax.legend(handles=handles)

    return ax


def _mean_and_error(values, error, axis=0):
    mean = values.mean(axis=axis)
    std = values.std(axis=axis, ddof=1)

    if error == "std":
        return mean, std

    if error == "se":
        return mean, std / np.sqrt(values.shape[axis])

    raise ValueError("error must be 'std' or 'se'.")


def plot_mean_errorbar(x, values, error="std", axis=0, ax=None):
    mean, errors = _mean_and_error(values, error, axis)

    if ax is None:
        _, ax = plt.subplots()

    ax.errorbar(
        x,
        mean,
        yerr=errors,
        marker="o",
        capsize=3,
    )

    return ax


def plot_mean_band(x, values, error="se", axis=0, ax=None):
    mean, errors = _mean_and_error(values, error, axis)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x, mean)
    ax.fill_between(
        x,
        mean - errors,
        mean + errors,
        alpha=0.2,
    )

    return ax


def _auto_bins(arrays):
    return np.histogram_bin_edges(
        np.concatenate(arrays),
        bins="auto",
    )


def plot_hists(
    data,
    ax=None,
    density=True,
    histtype="bar",
    ylabel=None,
    xlabel=None,
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots()

    if ylabel is None:
        ylabel = "Density" if density else "Count"

    bins = _auto_bins(list(data.values()))

    for label, array in data.items():
        ax.hist(
            array,
            bins=bins,
            density=density,
            histtype=histtype,
            label=label,
            **kwargs,
        )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.legend()
    return ax


def add_hline(ax, y=0.0, linewidth=0.8, linestyle="--", color="gray", **kwargs):
    for ax_ in np.atleast_1d(ax).ravel():
        ax_.axhline(
            y,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            **kwargs,
        )

    return ax
