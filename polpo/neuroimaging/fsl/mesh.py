import pyvista as pv

from polpo.dataset import Dataset
from polpo.neuroimaging._dispatch import read_geometry, select_mesh_paths

from .naming import get_all_subcortical_structs
from .validation import validate_structs


@read_geometry.register("fsl")
def read_geometry(path):
    """Read an FSL surface mesh.

    Parameters
    ----------
    path : path-like
        Path to a VTK surface mesh.

    Returns
    -------
    vertices : array-like, shape (n_vertices, 3)
        Mesh vertices.
    faces : array-like, shape (n_faces, 3)
        Triangular mesh faces.
    """
    mesh = pv.read(path)
    return mesh.points, mesh.regular_faces


@select_mesh_paths.register("fsl")
def select_mesh_paths(path, struct_subset=None):
    """Select FSL mesh paths by anatomical structure.

    Parameters
    ----------
    path : pathlib.Path
        Directory containing FSL FIRST outputs.
    struct_subset : array-like
        Structure identifiers to select. If ``None``, all subcortical
        structures are used.

    Returns
    -------
    mesh_paths : Dataset
        Mesh paths indexed by structure identifier.

    Raises
    ------
    ValueError
        If a structure identifier is invalid or if exactly one mesh cannot
        be found for a requested structure.
    """
    if struct_subset is None:
        struct_subset = get_all_subcortical_structs()

    validate_structs(struct_subset)

    mesh_paths = {}

    for struct in struct_subset:
        # e.g. sub-01_ses-01-L_Hipp_first.vtk
        matches = list(path.glob(f"*-{struct}_first.vtk"))

        if len(matches) != 1:
            raise ValueError(
                f"Expected one mesh for {struct!r} in {path!s}, "
                f"found {len(matches)}."
            )
        mesh_paths[struct] = matches[0]

    return Dataset(mesh_paths)
