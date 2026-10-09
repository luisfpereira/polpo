from polpo.dispatch import prefixdispatch


@prefixdispatch("derivative")
def select_mesh_paths(path, struct_subset=None, derivative="enigma"):
    """Select subcortical mesh files from a derivative directory.

    Parameters
    ----------
    path : path-like
        Directory containing the mesh files.
    struct_subset : iterable of str or None
        Anatomical structures to select. If None, select all available
        structures.
    derivative : str
        Derivative defining the mesh naming conventions.

    Returns
    -------
    Dataset
        Mapping from anatomical structures to mesh file paths.

    Raises
    ------
    ValueError
        If the derivative is not supported.
    """
    raise ValueError(f"Unknown derivative: {derivative}")


@prefixdispatch("derivative")
def read_geometry(path, derivative="enigma"):
    """Read mesh geometry from a derivative-specific file.

    Parameters
    ----------
    path : path-like
        Path to the mesh file.
    derivative : str
        Derivative defining the mesh file format.

    Returns
    -------
    vertices : array-like, shape=[n_vertices, 3]
        Vertex coordinates.
    faces : array-like, shape=[n_faces, 3]
        Triangular face indices.

    Raises
    ------
    ValueError
        If the derivative is not supported.
    """
    raise ValueError(f"Unknown derivative: {derivative}")


@prefixdispatch("derivative")
def select_segmentation_path(path, derivative="fast", **kwargs):
    """Select a subcortical segmentation file.

    Parameters
    ----------
    path : path-like
        Directory containing the segmentation outputs.
    derivative : str
        Segmentation derivative to select.
    **kwargs
        Arguments forwarded to the derivative-specific selector.

    Returns
    -------
    pathlib.Path
        Path to the selected segmentation file.
    """
    raise ValueError(f"Unknown derivative: {derivative}")


@prefixdispatch("derivative")
def compute_segmentation_volumes(path, labels, derivative="fast"):
    """Compute structure volumes from a derivative-specific segmentation."""
    raise ValueError(f"Unknown derivative: {derivative}")
