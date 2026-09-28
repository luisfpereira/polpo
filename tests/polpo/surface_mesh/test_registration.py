import numpy as np
import pytest

from polpo.surface_mesh.core import Surface
from polpo.surface_mesh.generation.blob import create_blob
from polpo.surface_mesh.registration import SurfaceRigidRegistration
from polpo.testing.data import LazyCaseData
from polpo.testing.lazy import LazyValue
from polpo.testing.parametrizers import DataBasedParametrizer


class SurfaceRigidRegistrationTestCase:
    @pytest.mark.sanity
    def test_fit_transform_preserves_faces(self, source):
        result = self.registration.fit_transform(source, source)
        np.testing.assert_array_equal(result.faces, source.faces)

    @pytest.mark.sanity
    def test_fit_transform_consistent(self, source, target, atol=1e-6):
        expected = self.registration.fit(source, target).transform(source)
        result = self.registration.fit_transform(source, target)

        np.testing.assert_array_equal(result.faces, expected.faces)
        np.testing.assert_allclose(result.vertices, expected.vertices, atol=atol)


class SurfaceRigidRegistrationTestData(LazyCaseData):
    def __init__(self, resolution=10):
        super().__init__()
        self.resolution = resolution

    def _make_surface(self):
        polydata = create_blob(resolution=self.resolution)
        return Surface.from_polydata(polydata)

    def fit_transform_preserves_faces_test_data(self):
        return [(LazyValue(self._make_surface),)]

    def fit_transform_consistent_test_data(self):
        return [(LazyValue(self._make_surface), LazyValue(self._make_surface))]


@pytest.fixture(
    scope="class",
    params=[True, False],
)
def registration_algos(request):
    known_correspondences = request.param

    request.cls.registration = SurfaceRigidRegistration(
        known_correspondences=known_correspondences
    )


@pytest.mark.usefixtures("registration_algos")
class TestSurfaceRigidRegistration(
    SurfaceRigidRegistrationTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = SurfaceRigidRegistrationTestData()
