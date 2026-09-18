import geomstats.backend as gs
import numpy as np

from polpo.surface_mesh.varifold.tuning import SigmaFromScale
from polpo.testing.data import DataCase
from polpo.testing.parametrizers import DataBasedParametrizer


class SigmaFromScaleTestCase:
    def test_sigma(self, surfaces, policy, expected, atol=gs.atol):
        sigma = policy(surfaces)

        np.testing.assert_allclose(sigma, expected, atol=atol)


class SigmaFromScaleTestData(DataCase):
    def sigma_test_data(self):
        return [
            (
                [1.0, 2.0],
                SigmaFromScale(
                    object_scale=lambda surface: surface,
                    discretization_scale=lambda surface: 1.0,
                    object_ratio=2.0,
                    discretization_ratio=1.0,
                ),
                4.0,
            ),
            (
                [1.0, 2.0],
                SigmaFromScale(
                    object_scale=lambda surface: 1.0,
                    discretization_scale=lambda surface: surface,
                    object_ratio=1.0,
                    discretization_ratio=2.0,
                ),
                4.0,
            ),
        ]


class TestSigmaFromScale(SigmaFromScaleTestCase, metaclass=DataBasedParametrizer):
    testing_data = SigmaFromScaleTestData()
