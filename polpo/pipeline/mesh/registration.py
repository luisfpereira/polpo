import numpy as np

import polpo.preprocessing.dict as ppdict
from polpo.preprocessing._preprocessing import Map, PartiallyInitializedStep
from polpo.preprocessing.base import PreprocessingStep
from polpo.preprocessing.mesh.adapter import PointCloudAdapter
from polpo.preprocessing.point_cloud.registration import (
    CorrespondenceBasedRigidAlignment,
)

try:
    from polpo.preprocessing._pyvista import PvAlign  # noqa:F401
except ImportError:
    pass

try:
    from polpo.preprocessing._h2_surfacematch import H2MeshAligner  # noqa:F401
except ImportError:
    pass

try:
    from polpo.preprocessing._skshapes import SksRigidRegistration  # noqa:F401
except ImportError:
    pass


class IdentityMeshAligner(PreprocessingStep):
    # useful for debugging

    def __call__(self, meshes):
        target_mesh, _ = meshes
        return target_mesh


class RigidAlignment(PartiallyInitializedStep):
    def __init__(self, target=None, known_correspondences=False, **kwargs):
        def _data2iter(data):
            if isinstance(data, dict):
                return ppdict.DictMap

            if isinstance(data, (list, tuple)):
                return Map

            return lambda x: x

        if target is None:

            def target(data):
                if isinstance(data, dict):
                    return next(iter(data.values()))

                return data[0]

        if known_correspondences:
            _Step = self._init_known_correspondences()
        else:
            _Step = self._init_unknown_correspondences()

        super().__init__(Step=_Step, _target=target, _data2iter=_data2iter, **kwargs)

    def _init_known_correspondences(self):
        def _Step(target, data2iter, **kwargs):
            # TODO: implement similar to `register_vertices_attr`?
            template_faces = np.array(target.faces).reshape(-1, 4)[:, 1:]

            mesh2points, points2mesh = PointCloudAdapter.build_pipes(template_faces)
            step = PointCloudAdapter(
                CorrespondenceBasedRigidAlignment(target=mesh2points(target)),
                mesh2points,
                points2mesh,
            )

            return data2iter(step)

        return _Step

    def _init_unknown_correspondences(self):
        def _Step(target, data2iter, **kwargs):
            step = PvAlign(target, **kwargs)
            return data2iter(step)

        return _Step
