from polpo.neuroimaging.dataset import group_by_structure
from polpo.neuroimaging.mesh import read_geometry, select_mesh_paths
from polpo.surface_mesh import Surface

from .defaults import DATA_DIR
from .path import select_folders


def load_dataset(
    derivative,
    data_dir=None,
    subject_subset=None,
    session_subset=None,
    struct_subset=None,
    as_surface=False,
):
    """Load derivative meshes grouped by structure.

    Parameters
    ----------
    derivative : str
        Prefix identifying the derivative directory.
    data_dir : str or pathlib.Path
        Dataset root directory.
    subject_subset : array-like
        Subject identifiers to select. If ``None``, all subjects are used.
    session_subset : array-like
        Session identifiers to select. If ``None``, all sessions are used.
    struct_subset : array-like
        Structure identifiers to select. If ``None``, all structures are used.
    as_surface : bool
        Whether to load mesh paths as ``Surface`` objects.

    Returns
    -------
    dataset : Dataset
        Dataset indexed by structure. Each value is a ``NestedDataset`` indexed
        by subject and session.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    folders = select_folders(
        data_dir,
        derivative,
        subject_subset=subject_subset,
        session_subset=session_subset,
    )

    def _select_mesh_paths(path):
        paths = select_mesh_paths(
            path, struct_subset=struct_subset, derivative=derivative
        )

        if as_surface:
            return paths.map_values(
                lambda path: Surface(*read_geometry(path, derivative=derivative))
            )

        return paths

    dataset = folders.map_values(_select_mesh_paths)

    return group_by_structure(dataset)
