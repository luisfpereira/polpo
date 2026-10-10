from matplotlib import pyplot as plt


def plot_distmat(dists, ax=None, title=None, show_ticks=True, **kwargs):
    """Plot a pairwise distance matrix.

    Parameters
    ----------
    dists : PairwiseDistances
        Pairwise distances to visualize.
    ax : matplotlib.axes.Axes
        Axes on which to plot. A new figure is created if None.
    title : str
        Plot title.
    show_ticks : bool
        Whether to display sample labels on both axes.
    **kwargs
        Keyword arguments passed to ``ax.imshow``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the distance matrix.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(dists.matrix, **kwargs)

    if title is not None:
        ax.set_title(title)

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

        return ax

    labels = dists.labels
    ticks = range(len(labels))

    ax.set_xticks(ticks, labels=labels, rotation=90)
    ax.set_yticks(ticks, labels=labels)

    return ax


def _group_pairs(pairs, grouper):
    """Group label pairs according to a callable applied to each pair."""
    groups = {}

    for pair in pairs:
        group = grouper(*pair)
        groups.setdefault(group, []).append(pair)

    return groups


def plot_dist_comparison(
    xdist,
    ydist,
    group_by=None,
    colors=None,
    xlabel="Local distance",
    ylabel="Global distance",
    ax=None,
    identity_line=True,
    rasterized=True,
    **kwargs,
):
    """Compare two collections of pairwise distances in a scatter plot.

    Both collections must have the same ordered labels.

    Parameters
    ----------
    xdist : PairwiseDistances or PairDistances
        Distances plotted on the horizontal axis.
    ydist : PairwiseDistances or PairDistances
        Distances plotted on the vertical axis.
    group_by : callable
        Function taking two sample labels and returning a group identifier.
    colors : dict
        Mapping from group identifiers to colors.
    xlabel : str
        Horizontal axis label.
    ylabel : str
        Vertical axis label.
    ax : matplotlib.axes.Axes
        Axes on which to plot. A new figure is created if None.
    identity_line : bool
        Whether to draw the identity line and use equal axis limits.
    rasterized : bool
        Whether to rasterize grouped scatter points.
    **kwargs
        Keyword arguments passed to ``ax.scatter`` for grouped points.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the distance comparison.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if xdist.labels != ydist.labels:
        raise ValueError("Not same key order!")

    x = xdist.data
    y = ydist.data

    if group_by is None or colors is None:
        ax.scatter(
            x,
            y,
            rasterized=rasterized,
            **kwargs,
        )
    else:
        groups = _group_pairs(xdist.pairs, group_by)

        for category, pairs in groups.items():
            x_group = xdist.select_pairs(pairs)
            y_group = ydist.select_pairs(pairs)

            ax.scatter(
                x_group.data,
                y_group.data,
                color=colors[category],
                label=category,
                rasterized=rasterized,
                **kwargs,
            )

        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if identity_line:
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]

        ax.plot(lims, lims, "--", color="gray")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    return ax
