import os

import meshio

from polpo.preprocessing.base import PreprocessingStep


class MeshioReader(PreprocessingStep):
    def __call__(self, path):
        return meshio.read(path)


class MeshioWriter(PreprocessingStep):
    def __init__(self, ext=None, dirname="", **kwargs):
        self.dirname = dirname
        self.ext = ext
        self.kwargs = kwargs

    def __call__(self, data):
        # TODO: maybe do opposite instead (give preference to arguments)
        # filename extension ignored if ext is not None
        filename, mesh = data
        if self.ext is not None:
            if "." in filename:
                filename = filename.split(".")[0]
            filename += f".{self.ext}"

        path = os.path.join(self.dirname, filename)
        mesh.write(path, **self.kwargs)

        return path


class MeshioFromData(PreprocessingStep):
    def __init__(self, cell_type="triangle"):
        super().__init__()
        self.cell_type = cell_type

    def __call__(self, mesh):
        # TODO: also consider colors
        if len(mesh) == 3:
            vertices, faces, _ = mesh
        else:
            vertices, faces = mesh
            _ = None
        return meshio.Mesh(
            points=vertices,
            cells=[meshio.CellBlock(cell_type=self.cell_type, data=faces)],
        )
