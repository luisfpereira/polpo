import numpy as np

from polpo.pipeline.base import PreprocessingStep


class TrimeshFaceRemoverByArea(PreprocessingStep):
    # TODO: generalize?

    def __init__(self, threshold=0.01, inplace=True):
        super().__init__()
        self.threshold = threshold
        self.inplace = inplace

    def __call__(self, mesh):
        if not self.inplace:
            mesh = mesh.copy()

        face_mask = ~np.less(mesh.area_faces, self.threshold)
        mesh.update_faces(face_mask)

        return mesh


class TrimeshDegenerateFacesRemover(PreprocessingStep):
    """Trimesh degenerate faces remover.

    https://trimesh.org/trimesh.base.html#trimesh.base.Trimesh.nondegenerate_faces

    Parameters
    ----------
    height: float
        Identifies faces with an oriented bounding box shorter than
        this on one side.
    """

    def __init__(self, height=1e-08, inplace=True):
        super().__init__()
        self.height = height
        self.inplace = inplace

    def __call__(self, mesh):
        if not self.inplace:
            mesh = mesh.copy()

        faces = mesh.nondegenerate_faces(height=self.height)
        mesh.update_faces(faces)
        return mesh


class TrimeshLargestComponentSelector(PreprocessingStep):
    def __init__(self, only_watertight=False):
        super().__init__()
        self.only_watertight = only_watertight

    def __call__(self, mesh):
        components = mesh.split(only_watertight=self.only_watertight)
        if len(components) == 0:
            return mesh

        components.sort(key=lambda component: len(component.faces), reverse=True)
        return components[0]
