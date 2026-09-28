import numpy as np

from polpo.registration.rigid import RigidRegistration
from polpo.testing.data import CaseData
from polpo.testing.parametrizers import DataBasedParametrizer


def apply_rigid_motion(source, translation, rotation):
    """Apply a rigid motion to a point cloud.

    Parameters
    ----------
    source : array-like, shape (n_points, 3)
        Source point cloud.
    translation : array-like, shape (3,)
        Translation vector.
    rotation : array-like, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    transformed : ndarray, shape (n_points, 3)
        Transformed point cloud.
    """
    return source @ rotation.T + translation


class RigidRegistrationTestCase:
    def test_fit_transform(self, source, target, expected, atol=1e-6):
        result = self.registration.fit_transform(source, target)

        np.testing.assert_allclose(result, expected, atol=atol)

    def test_fit_transform_recovers_target(
        self, source, rotation, translation, atol=1e-6
    ):
        target = apply_rigid_motion(source, translation, rotation)
        return self.test_fit_transform(source, target, target, atol=atol)

    def test_fit_corrects_reflection(self, source, reflection, atol=1e-6):
        target = source @ reflection.T

        self.registration.fit(source, target)
        det = np.linalg.det(self.registration.transform_[:3, :3])

        np.testing.assert_allclose(det, 1.0, atol=atol)


class RigidRegistrationTestData(CaseData):
    def _simple_point_cloud(self):
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )

    def fit_transform_recovers_target_test_data(self):
        point_cloud = self._simple_point_cloud()
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.array([1.0, 2.0, 3.0])

        return [(point_cloud, rotation, translation)]

    def fit_corrects_reflection_test_data(self):
        source = self._simple_point_cloud()
        reflections = [
            np.diag([-1.0, 1.0, 1.0]),
            np.diag([1.0, -1.0, 1.0]),
            np.diag([1.0, 1.0, -1.0]),
        ]

        return [(source, r) for r in reflections]


class TestRigidRegistration(
    RigidRegistrationTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = RigidRegistrationTestData()

    registration = RigidRegistration()
