from matplotlib import pyplot as plt


def get_colors(keys, cmap="tab20"):
    keys = list(keys)
    colors = plt.get_cmap(cmap).resampled(len(keys))

    return {key: colors(i) for i, key in enumerate(keys)}


def plot_nested(
    data,
    ax=None,
    colors=None,
    include_label=False,
    kind="plot",
    xlabel=None,
    ylabel=None,
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots()

    if colors is None:
        colors = get_colors(data.keys())

    plot = getattr(ax, kind)

    for outer_key, outer_data in data.iter_outer():
        plot(
            outer_data.keys_list(),
            outer_data.values_list(),
            color=colors[outer_key],
            label=outer_key if include_label else None,
            **kwargs,
        )

    if include_label:
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax
