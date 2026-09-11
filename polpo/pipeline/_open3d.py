import numpy as np
import open3d as o3d

from polpo.preprocessing.base import PreprocessingStep, RegistrationStep


class O3dPointCloudFromNp(PreprocessingStep):
    def __call__(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd


class O3dIcp(RegistrationStep):
    def __init__(
        self, target=None, threshold=None, estimation_method=None, return_matrix=False
    ):
        super().__init__(target=target)
        if estimation_method is None:
            estimation_method = (
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
        self.threshold = threshold
        self.estimation_method = estimation_method
        self.return_matrix = return_matrix

    def _threshold(self, source, target):
        return (
            np.amax(
                np.linalg.norm(
                    np.array(source.points) - np.array(target.points),
                    axis=-1,
                )
            )
            * 1.2
        )

    def __call__(self, data):
        source, target = self._get_source_and_target(data)

        threshold = self.threshold or self._threshold(source, target)

        reg_res = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            estimation_method=self.estimation_method,
        )

        # TODO: done in place; confirm if desired behavior
        source.transform(reg_res.transformation)
        if not self.return_matrix:
            return source

        return source, reg_res.transformation
