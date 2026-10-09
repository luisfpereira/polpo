from polpo.neuroimaging._dispatch import select_segmentation_path as _dispatcher


@_dispatcher.register("fsl")
def select_segmentation_path(path):
    """Select an FSL FIRST subcortical segmentation file.

    Parameters
    ----------
    path : path-like
        Directory containing FSL FIRST outputs.

    Returns
    -------
    pathlib.Path
        Path to the combined subcortical segmentation file.

    Raises
    ------
    FileNotFoundError
        If no matching segmentation file is found.
    ValueError
        If multiple matching segmentation files are found.
    """
    paths = list(path.rglob("*all_fast_firstseg*.nii.gz"))

    if not paths:
        raise FileNotFoundError(f"No FSL segmentation found in {path}.")

    if len(paths) > 1:
        raise ValueError(f"Multiple FSL segmentations found in {path}: {paths}")

    return paths[0]
