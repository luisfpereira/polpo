import geomstats.backend as gs
import numpy as np

from polpo.surface_mesh import Surface
from polpo.surface_mesh.transform import DeltaTransform
from polpo.testing.data import CaseData
from polpo.testing.parametrizers import DataBasedParametrizer


class InvertibleTransformTestCase:
    def assert_image_equal(self, result, expected, atol):
        np.testing.assert_allclose(result, expected, atol=atol)

    def assert_domain_equal(self, result, expected, atol):
        np.testing.assert_allclose(result, expected, atol=atol)

    def test_transform(self, point, expected, atol=gs.atol):
        result = self.transform(point)

        self.assert_image_equal(result, expected, atol)

    def test_inverse_after_transform(self, point, atol=gs.atol):
        result = self.transform.inverse(self.transform(point))

        self.assert_domain_equal(result, point, atol)


class DeltaTransformTestData(CaseData):
    def __init__(self, template=None):
        super().__init__()

        if template is None:
            template = Surface(
                vertices=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ]
                ),
                faces=np.array([[0, 1, 2]]),
            )

        self.template = template

    def transform_test_data(self):
        surface = Surface(
            vertices=self.template.vertices
            + np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                ]
            ),
            faces=self.template.faces,
        )

        expected = np.arange(1.0, 10.0)

        return [(surface, expected)]

    def inverse_after_transform_test_data(self):
        surface, _ = self.transform_test_data()[0]
        return [(surface,)]


class TestDeltaTransform(InvertibleTransformTestCase, metaclass=DataBasedParametrizer):
    testing_data = DeltaTransformTestData()

    transform = DeltaTransform(testing_data.template)

    def assert_domain_equal(self, result, expected, atol):
        np.testing.assert_allclose(result.vertices, expected.vertices, atol=atol)
        np.testing.assert_array_equal(result.faces, expected.faces)
