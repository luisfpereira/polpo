import os
from pathlib import Path

import pyvista as pv

from polpo.pipeline.base import PreprocessingStep


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


class PvWriter(PreprocessingStep):
    """Write a surface mesh to disk.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata.save
    """

    def __init__(
        self,
        dirname="",
        ext=None,
        binary=True,
        recompute_normals=False,
        exists_ok=True,
    ):
        super().__init__()
        self.dirname = dirname
        self.ext = ext
        self.exists_ok = exists_ok

        self.binary = binary
        self.recompute_normals = recompute_normals

    def __call__(self, data):
        """Apply step.

        Parameters
        ----------
        data : tuple[str, pv.PolyData]
            Filename and mesh to write.

        Returns
        -------
        path : str
            Filename.
        """
        filename, poly_data = data

        if "." not in Path(filename).name and self.ext is not None:
            filename += f".{self.ext}"

        ext = filename.split(".")[1]

        path = Path(os.path.join(self.dirname, filename))
        path.parent.mkdir(parents=True, exist_ok=self.exists_ok)

        texture = (
            poly_data["colors"]
            if ext == "ply" and "colors" in poly_data.array_names
            else None
        )

        poly_data.save(
            path,
            binary=self.binary,
            texture=texture,
            recompute_normals=self.recompute_normals,
        )

        return path
