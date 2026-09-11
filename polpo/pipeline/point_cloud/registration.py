try:
    from polpo.preprocessing._open3d import O3dIcp  # noqa:F401
except ImportError:
    pass

from polpo.preprocessing.base import RegistrationStep
from polpo.registration.point_cloud import kabsch


class CorrespondenceBasedRigidAlignment(RegistrationStep):
    def __init__(self, target=None, return_matrix=False):
        super().__init__(target=target)
        self.return_matrix = return_matrix

    def __call__(self, data):
        source, target = self._get_source_and_target(data)

        transform = kabsch(source, target, as_homogeneous=True)

        rotation_matrix = transform[:3, :3]
        translation = transform[:3, 3]

        aligned_source = (rotation_matrix @ source.T).T + translation
        if self.return_matrix:
            return aligned_source, transform

        return aligned_source
