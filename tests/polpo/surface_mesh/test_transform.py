import geomstats.backend as gs
import numpy as np

from polpo.surface_mesh.core import Surface
from polpo.surface_mesh.transform import DeltaTransform
from polpo.testing.data import CaseData
from polpo.testing.parametrizers import DataBasedParametrizer


class DeltaTransformTestData(CaseData):
    def transform_test_data(self):
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

        surface = Surface(
            vertices=template.vertices
            + np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                ]
            ),
            faces=template.faces,
        )

        expected = np.arange(1.0, 10.0)

        return [(template, surface, expected)]

    def inverse_after_transform_test_data(self):
        template, surface, _ = self.transform_test_data()[0]
        return [(template, surface)]


class TestDeltaTransform(metaclass=DataBasedParametrizer):
    testing_data = DeltaTransformTestData()

    def test_transform(self, template, surface, expected, atol=gs.atol):
        transform = DeltaTransform(template)

        result = transform(surface)

        np.testing.assert_allclose(result, expected, atol=atol)

    def test_inverse_after_transform(self, template, surface, atol=gs.atol):
        transform = DeltaTransform(template)

        result = transform.inverse(transform(surface))

        np.testing.assert_allclose(
            result.vertices,
            surface.vertices,
            atol=atol,
        )
        np.testing.assert_array_equal(result.faces, surface.faces)
