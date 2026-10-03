from matplotlib import pyplot as plt


def plot_dist_mat(dists, title=None, fig_size=None):
    # TODO: add as_method to PairwiseDists
    fig, ax = plt.subplots(figsize=fig_size)

    im = ax.imshow(dists.matrix)

    plt.colorbar(im)

    if title is not None:
        ax.set_title(title)

    keys = dists.labels
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90)

    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)

    return ax


def _group_pairs(pairs, grouper):
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
    if ax is None:
        fig, ax = plt.subplots()

    if xdist.labels != ydist.labels:
        raise ValueError("Not same key order!")

    x = xdist.data
    y = ydist.data

    if group_by is None or colors is None:
        ax.scatter(x, y)
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
