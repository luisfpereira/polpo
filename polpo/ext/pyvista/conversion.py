import numpy as np
import pyvista as pv

from polpo.preprocessing.base import PreprocessingStep


class PvFromData(PreprocessingStep):
    """Convert arrays into pv.PolyData.

    Parameters
    ----------
    array_names : list[str]
        Array names.
    """

    def __init__(self, array_names=()):
        super().__init__()
        self.array_names = array_names

    def __call__(self, mesh):
        """Apply step.

        Parameters
        ----------
        mesh : tuple
            Mesh represented by (vertices, faces, *arrays).

        Returns
        -------
        mesh : pv.PolyData
            Converted mesh.
        """
        vertices, faces, *arrays = mesh

        poly_data = pv.PolyData.from_regular_faces(points=vertices, faces=faces)
        for name, array in zip(self.array_names, arrays):
            poly_data[name] = array
        return poly_data


class DataFromPv(PreprocessingStep):
    """Convert pv.PolyData into arrays.

    Parameters
    ----------
    array_names : list[str]
        Array names. If None, return all as a dict.
    """

    def __init__(self, array_names=()):
        super().__init__()
        self.array_names = array_names

    def __call__(self, mesh):
        vertices, faces = (
            np.array(mesh.points),
            np.array(mesh.regular_faces),
        )

        if self.array_names is None:
            arrays = {name: np.array(mesh[name]) for name in mesh.array_names}
            return vertices, faces, arrays

        if len(self.array_names):
            arrays = [np.array(mesh[name]) for name in self.array_names]
            return vertices, faces, *arrays

        return vertices, faces
