import pyvista as pv

from polpo.surface_mesh.core import Surface


def write_vtk_polydata(path, surface):
    """Write a surface as legacy VTK PolyData."""
    surface.to_polydata().save(path)


def read_vtk_polydata(path):
    """Read a legacy VTK PolyData surface.

    Parameters
    ----------
    path : path-like
        Input VTK file path.

    Returns
    -------
    surface : Surface
        Surface mesh.
    """
    polydata = pv.read(path)
    return Surface.from_polydata(polydata)
