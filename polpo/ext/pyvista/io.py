import pyvista as pv

from polpo.preprocessing.base import PreprocessingStep


class PvReader(PreprocessingStep):
    """Read file.

    https://docs.pyvista.org/api/utilities/_autosummary/pyvista.read
    """

    def __init__(self, rename_colors=True):
        super().__init__()
        self.rename_colors = rename_colors

    def __call__(self, filename):
        """Apply step.

        Parameters
        ----------
        filename : str
            File name.

        Returns
        -------
        mesh : pv.PolyData
            Loaded mesh.
        """
        poly_data = pv.read(filename)

        if "RGBA" in poly_data.array_names:
            poly_data.rename_array("RGBA", "colors")
        elif "RGB" in poly_data.array_names:
            poly_data.rename_array("RGB", "colors")

        return poly_data
