"""Tests for surface mesh scale estimators."""

import geomstats.backend as gs
import numpy as np

from polpo.surface_mesh.core import Surface
from polpo.surface_mesh.generation.polyhedra import create_regular_tetrahedron
from polpo.surface_mesh.scale import (
    median_edge_length,
    surface_centroid_to_farthest_vertex,
    vertex_centroid_to_farthest_vertex,
)
from polpo.testing.data import DataCase, LazyValue
from polpo.testing.decorators import materialize_lazy_values
from polpo.testing.parametrizers import DataBasedParametrizer


class ScaleMethodsTestCase:
    """Test surface mesh scale estimators."""

    def test_median_edge_length(self, surface, expected, atol=gs.atol):
        value = median_edge_length(surface)
        np.testing.assert_allclose(value, expected, atol=atol)

    def test_vertex_centroid_to_farthest_vertex(self, surface, expected, atol=gs.atol):
        value = vertex_centroid_to_farthest_vertex(surface)
        np.testing.assert_allclose(value, expected, atol=atol)

    def test_surface_centroid_to_farthest_vertex(self, surface, expected, atol=gs.atol):
        value = surface_centroid_to_farthest_vertex(surface)
        np.testing.assert_allclose(value, expected, atol=atol)


class RegularTetrahedronTestData(DataCase):
    """Test data based on a regular tetrahedron."""

    def __init__(self, edge_length=2.0):
        super().__init__()
        self.edge_length = edge_length

        self.surface = LazyValue(
            lambda edge_length: Surface(*create_regular_tetrahedron(edge_length)),
            edge_length=edge_length,
        )

    def get_decorators(self):
        return [materialize_lazy_values]

    def median_edge_length_test_data(self):
        expected = self.edge_length
        return [(self.surface, expected)]

    def vertex_centroid_to_farthest_vertex_test_data(self):
        expected = gs.sqrt(6.0) / 4.0 * self.edge_length
        return [(self.surface, expected)]

    def surface_centroid_to_farthest_vertex_test_data(self):
        return self.vertex_centroid_to_farthest_vertex_test_data()


class TestScaleMethodsWithTetra(ScaleMethodsTestCase, metaclass=DataBasedParametrizer):
    """Test scale estimators on a regular tetrahedron."""

    testing_data = RegularTetrahedronTestData()
