import os

import trimesh


class TrimeshToPly:
    def __init__(
        self,
        dirname="",
        encoding="binary",
        vertex_normal=None,
        include_attributes=True,
    ):
        super().__init__()
        # TODO: create dir if does not exist?
        self.dirname = dirname
        self.encoding = encoding
        self.vertex_normal = vertex_normal
        self.include_attributes = include_attributes

        # TODO: add override?

    def __call__(self, data):
        filename, mesh = data

        ext = ".ply"
        if not filename.endswith(ext):
            filename += ext

        path = os.path.join(self.dirname, filename)

        ply_text = trimesh.exchange.ply.export_ply(
            mesh, encoding=self.encoding, include_attributes=self.include_attributes
        )

        with open(path, "wb") as file:
            file.write(ply_text)

        return path


class TrimeshReader:
    """Read file.

    Uses `load_mesh` (
    https://trimesh.org/trimesh.exchange.load.html#trimesh.exchange.load.load_mesh
    ) if supported format, which is very limited.

    If not supported format, goes through `pyvista`.
    Particularly relevant for `vtk` `Polydata`,
    otherwise `meshio` could have been used.
    """

    def __init__(self):
        super().__init__()
        self._supported_fmts = {"stl", "ply", "dxf"}

    def __call__(self, path):
        ext = path.split(".")[-1]
        if ext in self._supported_fmts:
            return trimesh.load_mesh(path)

        raise ValueError(f"``ext`` not supported.")
