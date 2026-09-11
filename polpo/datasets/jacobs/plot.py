import matplotlib as mpl
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt


def update_mpl_params(format="pdf"):
    mpl.rcParams.update(
        {
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.0,
            "savefig.format": "pdf",
            "image.cmap": "bwr",
        }
    )


def get_subject_colors(subj_ids):
    color_ids = [
        subj_id for subj_id in subj_ids if not str(subj_id).startswith(("3", "4"))
    ]

    tab10 = plt.colormaps["tab10"]
    base_colors = [tab10(i) for i in range(tab10.N) if i != 7]

    cmap = mcolors.ListedColormap(base_colors).resampled(len(color_ids))

    colors = dict(zip(color_ids, cmap(range(len(color_ids)))))
    colors.update(
        {subj_id: "gray" for subj_id in subj_ids if str(subj_id).startswith(("3", "4"))}
    )
    return colors
