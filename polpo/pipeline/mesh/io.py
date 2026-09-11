try:
    from polpo.preprocessing._trimesh import TrimeshReader, TrimeshToPly  # noqa:F401
except ImportError:
    pass

try:
    from polpo.preprocessing._meshio import MeshioReader, MeshioWriter  # noqa:F401
except ImportError:
    pass

try:
    from polpo.preprocessing._pyvista import PvReader, PvWriter  # noqa:F401
except ImportError:
    pass

try:
    from polpo.freesurfer.mesh import FreeSurferReader  # noqa:F401
except ImportError:
    pass

from polpo.preprocessing.base import PreprocessingStep


class DictMeshWriter(PreprocessingStep):
    def __init__(self, mesh_writer=None, key2name=None, **kwargs):
        super().__init__()

        if mesh_writer is None:
            mesh_writer = PvWriter(**kwargs)

        if key2name is None:
            key2name = lambda key: f"mesh_{key}"

        self.mesh_writer = mesh_writer
        self.key2name = key2name

    def __call__(self, data):
        return {
            key: self.mesh_writer((self.key2name(key), mesh))
            for key, mesh in data.items()
        }
